#import all necessary libraries
import os
import uuid
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, url_for
from werkzeug.utils import secure_filename
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.cm as cm

# Configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "Models")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "static", "uploads")
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}
IMG_SIZE = 224
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

app = Flask(__name__, template_folder=os.path.join(BASE_DIR, "templates"),
            static_folder=os.path.join(BASE_DIR, "static"))
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16MB max upload

os.makedirs(UPLOAD_FOLDER, exist_ok=True)


# Model Definitions

class CNN(nn.Module):
    def __init__(self):
        super(CNN, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(),
            nn.MaxPool2d(2, 2), nn.Dropout(0.25),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * (IMG_SIZE // 8) * (IMG_SIZE // 8), 256),
            nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
            nn.Linear(256, 1),
        )

    def forward(self, x):
        return self.classifier(self.features(x))


class PatchEmbedding(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=128):
        super().__init__()
        self.num_patches = (img_size // patch_size) ** 2
        self.projection = nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size)
        self.position_embedding = nn.Parameter(torch.randn(1, self.num_patches, embed_dim))

    def forward(self, x):
        x = self.projection(x)
        x = x.flatten(2).transpose(1, 2)
        x = x + self.position_embedding
        return x


class TransformerBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(embed_dim, num_heads, dropout=dropout, batch_first=True)
        self.norm2 = nn.LayerNorm(embed_dim)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        x_norm = self.norm1(x)
        attn_out, _ = self.attn(x_norm, x_norm, x_norm)
        x = x + attn_out
        x = x + self.mlp(self.norm2(x))
        return x


class VisionTransformer(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_channels=3, embed_dim=128,
                 num_heads=4, num_layers=4, mlp_dim=256, dropout=0.1):
        super().__init__()
        self.patch_embed = PatchEmbedding(img_size, patch_size, in_channels, embed_dim)
        self.transformer = nn.Sequential(
            *[TransformerBlock(embed_dim, num_heads, mlp_dim, dropout) for _ in range(num_layers)]
        )
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.3)
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 256), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(256, 128), nn.GELU(), nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, x):
        x = self.patch_embed(x)
        x = self.transformer(x)
        x = self.norm(x)
        x = x.mean(dim=1)
        x = self.dropout(x)
        return self.head(x)


def build_resnet50():
    model = models.resnet50(weights=None)
    num_features = model.fc.in_features
    model.fc = nn.Sequential(
        nn.Linear(num_features, 256), nn.ReLU(), nn.BatchNorm1d(256), nn.Dropout(0.5),
        nn.Linear(256, 128), nn.ReLU(), nn.BatchNorm1d(128), nn.Dropout(0.3),
        nn.Linear(128, 1),
    )
    return model


# Model Loading
IMAGENET_TRANSFORM = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

MODEL_CONFIGS = {
    "CNN": {"path": os.path.join(MODELS_DIR, "cifake_cnn_model.pth"), "builder": CNN, "transform": IMAGENET_TRANSFORM},
    "ResNet50": {"path": os.path.join(MODELS_DIR, "cifake_resnet50_model.pth"), "builder": build_resnet50, "transform": IMAGENET_TRANSFORM},
    "ViT": {"path": os.path.join(MODELS_DIR, "cifake_vit_model.pth"), "builder": VisionTransformer, "transform": IMAGENET_TRANSFORM},
}

loaded_models = {}


def load_models():
    for name, config in MODEL_CONFIGS.items():
        if os.path.exists(config["path"]):
            try:
                model = config["builder"]()
                model.load_state_dict(torch.load(config["path"], map_location=DEVICE, weights_only=True))
                model.to(DEVICE)
                model.eval()
                loaded_models[name] = model
                print(f"Loaded model: {name}")
            except Exception as e:
                print(f"Failed to load {name}: {e}")
        else:
            print(f"Model file not found: {config['path']}")


def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Explainability: Grad-CAM

def compute_gradcam_cnn(model, img_tensor):
    """Grad-CAM for the custom CNN (targets last conv layer in features block)."""
    # Hook to capture activations and gradients
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    # Find last Conv2d in features
    target_layer = None
    for module in model.features:
        if isinstance(module, nn.Conv2d):
            target_layer = module

    if target_layer is None:
        return None

    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    model.zero_grad()
    output.backward()

    fh.remove()
    bh.remove()

    # Compute Grad-CAM
    grads = gradients[0][0]          # (channels, h, w)
    acts = activations[0][0]         # (channels, h, w)
    weights = grads.mean(dim=(1, 2))  # Global average pool gradients
    heatmap = (weights[:, None, None] * acts).sum(dim=0)
    heatmap = torch.relu(heatmap)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap.detach().cpu().numpy()


def compute_gradcam_resnet(model, img_tensor):
    """Grad-CAM for ResNet50 (targets layer4)."""
    activations = []
    gradients = []

    def forward_hook(module, input, output):
        activations.append(output)

    def backward_hook(module, grad_input, grad_output):
        gradients.append(grad_output[0])

    target_layer = model.layer4[-1].conv3  # Last conv in layer4
    fh = target_layer.register_forward_hook(forward_hook)
    bh = target_layer.register_full_backward_hook(backward_hook)

    output = model(img_tensor)
    model.zero_grad()
    output.backward()

    fh.remove()
    bh.remove()

    grads = gradients[0][0]
    acts = activations[0][0]
    weights = grads.mean(dim=(1, 2))
    heatmap = (weights[:, None, None] * acts).sum(dim=0)
    heatmap = torch.relu(heatmap)
    heatmap = heatmap / (heatmap.max() + 1e-8)
    return heatmap.detach().cpu().numpy()


def compute_attention_rollout_vit(model, img_tensor):
    """Attention Rollout for the custom VisionTransformer (Abnar & Zuidema, 2020).
    Manually steps through each TransformerBlock to capture attention weights
    with need_weights=True. No CLS token — uses mean over all patch positions.
    """
    model.eval()
    with torch.no_grad():
        x = model.patch_embed(img_tensor)   # (1, 196, embed_dim)
        num_patches = x.shape[1]

        rollout = torch.eye(num_patches)

        for block in model.transformer:
            x_norm = block.norm1(x)
            attn_out, attn_weights = block.attn(
                x_norm, x_norm, x_norm,
                need_weights=True, average_attn_weights=True,
            )
            # attn_weights: (1, num_patches, num_patches)
            A = attn_weights.squeeze(0).cpu()   # (196, 196)
            A = A + torch.eye(num_patches)
            A = A / A.sum(dim=-1, keepdim=True)
            rollout = A @ rollout
            # Complete the block forward pass
            x = x + attn_out
            x = x + block.mlp(block.norm2(x))

    # No CLS token, average saliency across all patch positions
    attn_map = rollout.mean(dim=0).numpy()      # (196,)
    grid_size = int(num_patches ** 0.5)         # 14
    attn_map = attn_map.reshape(grid_size, grid_size)
    attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
    return attn_map


def generate_heatmap(model, model_name, img_tensor, original_image_path, base_name):
    """Generate and save a heatmap overlay."""
    try:
        if model_name == "CNN":
            heatmap = compute_gradcam_cnn(model, img_tensor)
        elif model_name == "ResNet50":
            heatmap = compute_gradcam_resnet(model, img_tensor)
        elif model_name == "ViT":
            heatmap = compute_attention_rollout_vit(model, img_tensor)
        else:
            return None

        if heatmap is None:
            return None

        # Load original image for overlay
        original_img = Image.open(original_image_path).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
        original_img = np.array(original_img) / 255.0

        # Resize heatmap to image size
        from PIL import Image as PILImage
        heatmap_pil = PILImage.fromarray((heatmap * 255).astype(np.uint8))
        heatmap_resized = np.array(heatmap_pil.resize((IMG_SIZE, IMG_SIZE))) / 255.0

        # Apply colormap
        colormap = cm.jet(heatmap_resized)[:, :, :3]

        # Overlay
        overlay = 0.6 * original_img + 0.4 * colormap
        overlay = np.clip(overlay, 0, 1)

        # Save
        heatmap_filename = f"{base_name}_cam_{model_name.lower()}.png"
        heatmap_path = os.path.join(UPLOAD_FOLDER, heatmap_filename)

        plt.figure(figsize=(4, 4))
        plt.imshow(overlay)
        plt.axis("off")
        plt.title(f"{model_name} - Activation Map", fontsize=10)
        plt.tight_layout(pad=0.5)
        plt.savefig(heatmap_path, dpi=100, bbox_inches="tight", pad_inches=0.1)
        plt.close()

        return heatmap_filename

    except Exception as e:
        print(f"Heatmap generation failed for {model_name}: {e}")
        return None


def predict_image(image_path):
    """Run the image through all loaded models and return results."""
    results = []
    base_name = os.path.splitext(os.path.basename(image_path))[0]

    for name, model in loaded_models.items():
        img = Image.open(image_path).convert("RGB")
        img_tensor = MODEL_CONFIGS[name]["transform"](img).unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            output = model(img_tensor).squeeze()
            prob = torch.sigmoid(output).item()

        label = "REAL" if prob > 0.5 else "AI-GENERATED"
        confidence = prob if prob > 0.5 else 1 - prob

        heatmap_filename = generate_heatmap(model, name, img_tensor.clone(), image_path, base_name)

        results.append({
            "model": name,
            "label": label,
            "confidence": float(confidence * 100),
            "raw_score": float(prob),
            "heatmap_url": url_for("static", filename=f"uploads/{heatmap_filename}") if heatmap_filename else None,
        })

    return results


# Routes
@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", results=None, image_url=None)


@app.route("/predict", methods=["POST"])
def predict():
    if "image" not in request.files:
        return render_template("index.html", results=None, image_url=None, error="No file uploaded.")

    file = request.files["image"]
    if file.filename == "":
        return render_template("index.html", results=None, image_url=None, error="No file selected.")

    if not allowed_file(file.filename):
        return render_template("index.html", results=None, image_url=None,
                               error="Invalid file type. Allowed: PNG, JPG, JPEG.")

    ext = secure_filename(file.filename).rsplit(".", 1)[1].lower()
    filename = f"{uuid.uuid4().hex}.{ext}"
    filepath = os.path.join(app.config["UPLOAD_FOLDER"], filename)
    file.save(filepath)

    results = predict_image(filepath)

    image_url = url_for("static", filename=f"uploads/{filename}")

    return render_template("index.html", results=results, image_url=image_url)


if __name__ == "__main__":
    load_models()
    if not loaded_models:
        print("\nWARNING: No models loaded. Ensure trained .pth files exist.")
        print("Expected files:")
        for name, config in MODEL_CONFIGS.items():
            print(f"  {config['path']}")
        print()
    app.run(debug=True, host="127.0.0.1", port=5000)
