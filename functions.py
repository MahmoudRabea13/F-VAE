# functions.py

import torch
from torchvision import transforms
from torchvision.utils import make_grid
from PIL import Image, ImageEnhance, ImageFilter
from facenet_pytorch import MTCNN



# === Load VAE Model ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = torch.load("pretrained_weights/vae_model_20.pth", map_location=device)
model.eval()

# === Face Detector ===
def detect_face_mtcnn(image, output_size=150):
    mtcnn = MTCNN(image_size=output_size, margin=20, post_process=True, device=device)
    img = image.convert('RGB')
    face = mtcnn(img)

    if face is None:
        raise ValueError("No face detected.")
    
    face = (face + 1) / 2
    face_img = transforms.ToPILImage()(face).convert("RGB")
    return face_img

# === Enhance Output ===
def enhance_image(tensor_img):
    img = transforms.ToPILImage()(tensor_img.cpu().clamp(0, 1))
    img = img.filter(ImageFilter.SMOOTH_MORE)
    img = ImageEnhance.Contrast(img).enhance(1.3)
    img = ImageEnhance.Sharpness(img).enhance(2.0)
    img = ImageEnhance.Color(img).enhance(1.2)
    return transforms.ToTensor()(img)

# === Latent Variation Generator ===
def generate_latents(model, img_tensor, method, num_samples, alpha, walk_scale, ramp_power):
    with torch.no_grad():
        mu, log_var = model.encode(img_tensor)
        z_list = []

        for i in range(num_samples):
            t = i / (num_samples - 1)
            step = (i - num_samples // 2) / (num_samples // 2)

            if method == 'noise_interp':
                z_rand = torch.randn_like(mu)
                z = (1 - alpha) * mu + alpha * z_rand
            elif method == 'latent_walk':
                z = mu + step * walk_scale
            elif method == 'noise_ramp':
                ramp = t ** ramp_power
                z_rand = torch.randn_like(mu)
                z = (1 - ramp) * mu + ramp * z_rand
            else:
                z = mu

            z_list.append(z)
        return torch.cat(z_list, dim=0)

# === Core Inference Function ===
def generate_faces(input_image, method, value):
    try:
        face_img = detect_face_mtcnn(input_image)
    except:
        return None, None, "❌ No face detected."

    transform = transforms.ToTensor()
    img_tensor = transform(face_img).unsqueeze(0).to(device)

    if method == "noise_interp":
        alpha = value
        walk_scale = 0
        ramp_power = 0
    elif method == "latent_walk":
        alpha = 0
        walk_scale = value
        ramp_power = 0
    elif method == "noise_ramp":
        alpha = 0
        walk_scale = 0
        ramp_power = value

    z_stack = generate_latents(model, img_tensor, method, 10, alpha, walk_scale, ramp_power)

    with torch.no_grad():
        outputs = model.decode(z_stack).view(-1, 3, 150, 150)
        outputs = torch.stack([enhance_image(img) for img in outputs])

    grid = make_grid(outputs, nrow=5)
    output_img = transforms.ToPILImage()(grid)

    return output_img, face_img, "✅ Face processed successfully!"


