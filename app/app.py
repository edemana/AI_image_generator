import streamlit as st
import torch
from PIL import Image


# Load the model
@st.cache_resource
def load_model():
    G = torch.load("full_model_stylegan.pt", map_location=torch.device("cpu"))
    return G


G = load_model()


def generate_images(G, num_images=1, truncation_psi=0.7, seed=None):
    if seed is not None:
        torch.manual_seed(seed)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    z = torch.randn((num_images, G.z_dim), device=device)

    ws = G.mapping(z, None, truncation_psi=truncation_psi)
    img = G.synthesis(ws, noise_mode="const")
    img = (
        (img.permute(0, 2, 3, 1) * 127.5 + 128)
        .clamp(0, 255)
        .to(torch.uint8)
        .cpu()
        .numpy()
    )

    return [Image.fromarray(i) for i in img]


st.title("Modified StyleGAN Image Generator")

num_images = st.slider("Number of images to generate", 1, 4, 1)
truncation_psi = st.slider("Truncation psi", 0.0, 1.0, 0.7)
seed = st.number_input("Random seed", value=None)

if st.button("Generate Images"):
    generated_images = generate_images(G, num_images, truncation_psi, seed)

    cols = st.columns(num_images)
    for i, img in enumerate(generated_images):
        cols[i].image(img, use_column_width=True)
