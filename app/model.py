# -*- coding: utf-8 -*-
"""Copy NGANof .ipynb

Automatically generated by Colab.

Original file is located at
    https://colab.research.google.com/drive/1w2sRg7uNq-lx67zg9Afcr58f2P_R-5ca
"""





# Import necessary libraries
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import os
import PIL.Image
import numpy as np
import dnnlib
import legacy

# Load pre-trained model
import pickle

with open('ffhq.pkl', 'rb') as f:
    data = pickle.load(f)

print(data.keys())

# Check if CUDA is available, and if so, move the model to GPU
if torch.cuda.is_available():
    G = data['G_ema'].cuda()
    D = data['D'].cuda()
    print("Model loaded on GPU.")
else:
    G = data['G_ema']  # Keep the model on CPU
    D = data['D'].cuda()
    print("CUDA not available, model loaded on CPU.")

print(type(G))
print(G)

print(type(D))
print(D)

def generate_images(G, z=None, num_images=1, truncation_psi=0.7, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    if torch.cuda.is_available():
        print("CUDA is available. Using GPU.")
    else:
        print("CUDA is not available. Using CPU.")

    if z is None:
        z = torch.randn((num_images, G.z_dim), device=device)
    else:
        z = z.to(device)

    print("Latent vectors prepared.")

    ws = G.mapping(z, None, truncation_psi=truncation_psi)
    print("Mapping done.")

    img = G.synthesis(ws, noise_mode='const')
    print("Synthesis done.")

    img = (img.permute(0, 2, 3, 1) * 127.5 + 128).clamp(0, 255).to(torch.uint8).cpu().numpy()
    print("Image tensor converted to numpy array.")

    return [Image.fromarray(i) for i in img]

from PIL import Image  # Add this import

# Generate 4 images
generated_images = generate_images(G, num_images=4, truncation_psi=0.7, seed=42)

# Generate and display 4 images
#generated_images = generate_images(G, num_images=4, truncation_psi=0.7, seed=42)
print("Images generated.")
for i, img in enumerate(generated_images):
    display(img)

# Fine-tuning setup
import os
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Advanced fine-tuning setup
import torch
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.utils import save_image

# Custom dataset
class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        self.image_dir = image_dir
        self.image_files = [f for f in os.listdir(image_dir) if f.endswith('.jpg') or f.endswith('.png')]

        if len(self.image_files) == 0:
            raise ValueError(f"No image files found in directory: {image_dir}")

        self.transform = transform

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        img_path = os.path.join(self.image_dir, self.image_files[idx])
        image = PIL.Image.open(img_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image

#!pip install datasets
from datasets import load_dataset
ds = load_dataset("TrainingDataPro/black-people-liveness-detection-video-dataset")

# Set up data loading
transform = transforms.Compose([
    transforms.Resize((G.img_resolution, G.img_resolution)),
    transforms.ToTensor(),
    transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
])

dataset = CustomDataset("/content/drive/MyDrive/Colab Notebooks/part2", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

# Fine-tuning loop (advanced)
optimizer_g = Adam(G.parameters(), lr=0.0001, betas=(0, 0.99), eps=1e-8)
optimizer_d = Adam(D.parameters(), lr=0.0001, betas=(0, 0.99), eps=1e-8)
scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.99)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.99)

num_epochs = 100
for epoch in range(num_epochs):
    for batch in dataloader:
        real_images = batch.cuda()

        # Generate fake images
        z = torch.randn([batch.shape[0], G.z_dim]).cuda()
        fake_images = G(z, None)  # Replace None with actual labels if available

        # Ensure fake_images requires gradients
        fake_images.requires_grad_(True)

        # Prepare a dummy label (replace with actual labels if available)
        c = None  # Replace 'some_dimension' with the correct size

        # Compute loss (using BCE loss)
        g_loss = torch.mean(torch.log(1 - D(fake_images, c)))  # Pass the dummy label to D
        d_loss_real = torch.mean(torch.log(D(real_images, c))) # Pass the dummy label to D
        d_loss_fake = torch.mean(torch.log(1 - D(fake_images, c))) # Pass the dummy label to D
        d_loss = -d_loss_real - d_loss_fake

        # Check for NaNs
        if torch.isnan(g_loss) or torch.isnan(d_loss):
            #print("NaN detected in loss. Skipping update.")
            continue

        # Update generator
        optimizer_g.zero_grad()
        g_loss.backward()
        torch.nn.utils.clip_grad_norm_(G.parameters(), max_norm=1)  # Clip gradients
        optimizer_g.step()

        # Update discriminator
        optimizer_d.zero_grad()
        d_loss.backward()
        torch.nn.utils.clip_grad_norm_(D.parameters(), max_norm=1)  # Clip gradients
        optimizer_d.step()

    # Update learning rate
    scheduler_g.step()
    scheduler_d.step()

    print(f"Epoch {epoch+1}/{num_epochs}, G Loss: {g_loss.item()}")

# Save the full model
torch.save(G, '/content/drive/MyDrive/full_model_stylegan.pt')

for param in G.parameters():
    param.requires_grad = True

# Generate new images with fine-tuned model
z = torch.randn([4, G.z_dim]).cuda()  # Generate 4 random latent vectors
imgs = generate_images(G, z, truncation_psi=0.7)

# Display each generated image
for img in imgs:
    display(img)

# Save the fine-tuned model
torch.save(G.state_dict(), 'fine_tuned_stylegan.pth')

#!pip install gradio

import torch
import torchvision.transforms as transforms
from PIL import Image
import gradio as gr
from tqdm import tqdm

def optimize_latent_vector(G, target_image, num_iterations=1000):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    target_image = transforms.Resize((G.img_resolution, G.img_resolution))(target_image)
    target_tensor = transforms.ToTensor()(target_image).unsqueeze(0).to(device)
    target_tensor = (target_tensor * 2) - 1  # Normalize to [-1, 1]

    latent_vector = torch.randn((1, G.z_dim), device=device, requires_grad=True)
    optimizer = torch.optim.Adam([latent_vector], lr=0.1)

    for i in tqdm(range(num_iterations), desc="Optimizing latent vector"):
        optimizer.zero_grad()

        generated_image = G(latent_vector, None)
        loss = torch.nn.functional.mse_loss(generated_image, target_tensor)

        loss.backward()
        optimizer.step()

        if (i + 1) % 100 == 0:
            print(f'Iteration {i+1}/{num_iterations}, Loss: {loss.item()}')

    return latent_vector.detach()

def generate_from_upload(uploaded_image):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Optimize latent vector for the uploaded image
    optimized_z = optimize_latent_vector(G, uploaded_image)

    # Generate variations
    num_variations = 4
    variation_strength = 0.1
    varied_z = optimized_z + torch.randn((num_variations, G.z_dim), device=device) * variation_strength

    # Generate the variations
    with torch.no_grad():
        imgs = G(varied_z, c=None, truncation_psi=0.7, noise_mode='const')

    imgs = (imgs * 127.5 + 128).clamp(0, 255).to(torch.uint8)
    imgs = imgs.permute(0, 2, 3, 1).cpu().numpy()

    # Convert the generated image tensors to PIL Images
    generated_images = [Image.fromarray(img) for img in imgs]

    # Return the images separately
    return generated_images[0], generated_images[1], generated_images[2], generated_images[3]

# Create the Gradio interface
iface = gr.Interface(
    fn=generate_from_upload,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil") for _ in range(4)],
    title="StyleGAN Image Variation Generator"
)

# Launch the Gradio interface
iface.launch(share=True, debug=True)

# If you want to test it without the Gradio interface:
"""
# Load an image explicitly
image_path = "path/to/your/image.jpg"
image = Image.open(image_path)

# Call the generate method explicitly
generated_images = generate_from_upload(image)

# Display the generated images
for img in generated_images:
    img.show()
"""

