# StyleGAN2 Fine-Tuning and Image Variation Generator

This project demonstrates the fine-tuning of a StyleGAN2 model on a custom dataset and provides a Gradio interface for generating image variations based on uploaded images. It utilizes the StyleGAN2-ADA-PyTorch implementation and includes advanced fine-tuning techniques.

## Features

- Loading and fine-tuning of a pre-trained StyleGAN2 model (FFHQ)
- Custom dataset loading for fine-tuning
- Advanced fine-tuning loop with gradient clipping and learning rate scheduling
- Image generation using the fine-tuned model
- Latent vector optimization for input images
- Gradio interface for uploading images and generating variations

## Installation

1. Clone the Nvidia's StyleGAN2-ADA-PyTorch repository:
https://github.com/NVlabs/stylegan2-ada-pytorch.git


2. Install the following required packages:
pip install ninja torch torchvision torchaudio tqdm gradio


3. Download the pre-trained FFHQ model:
wget https://nvlabs-fi-cdn.nvidia.com/stylegan2-ada-pytorch/pretrained/ffhq.pkl

4. Set up your Google Drive (if using Google Colab):
from google.colab import drive
drive.mount('/content/drive')


## Usage

1. Load Pre-trained Model:

import pickle

with open('ffhq.pkl', 'rb') as f:
    data = pickle.load(f)

G = data['G_ema'].cuda()  # Generator
D = data['D'].cuda()      # Discriminator



2. Fine-tune the Model:

a. Prepare your custom dataset:

• Adjust the CustomDataset class to point to your image directory

dataset = CustomDataset("/content/drive/MyDrive/Colab Notebooks/part2", transform=transform)
dataloader = DataLoader(dataset, batch_size=4, shuffle=True)

b. Run the fine-tuning loop:

num_epochs = 100
for epoch in range(num_epochs):
    for batch in dataloader:
        # Training loop code here



3. Generate Images

Use the generate_images function to create new images:

generated_images = generate_images(G, num_images=4, truncation_psi=0.7, seed=42)



4. Launch the Gradio Interface

Run the script to start the Gradio interface:

iface = gr.Interface(
    fn=generate_from_upload,
    inputs=gr.Image(type="pil"),
    outputs=[gr.Image(type="pil") for _ in range(4)],
    title="StyleGAN Image Variation Generator"
)
iface.launch(share=True, debug=True)



## Detailed Components

## Custom Dataset

Dataset Link: https://drive.google.com/drive/folders/1yWxH6dQHlBobnsV1RIFjbJf0CBmXW8md?usp=sharing

The CustomDataset class is used to load images from a specified directory:

class CustomDataset(Dataset):
    def __init__(self, image_dir, transform=None):
        # Implementation details here



## Fine-tuning Setup

The fine-tuning process uses Adam optimizer with custom learning rates and betas:

optimizer_g = Adam(G.parameters(), lr=0.0001, betas=(0, 0.99), eps=1e-8)
optimizer_d = Adam(D.parameters(), lr=0.0001, betas=(0, 0.99), eps=1e-8)

Learning rate scheduling is implemented using ExponentialLR:

scheduler_g = torch.optim.lr_scheduler.ExponentialLR(optimizer_g, gamma=0.99)
scheduler_d = torch.optim.lr_scheduler.ExponentialLR(optimizer_d, gamma=0.99)



## Latent Vector Optimization

The optimize_latent_vector function is used to find the optimal latent vector for an input image:

def optimize_latent_vector(G, target_image, num_iterations=1000):
    # Implementation details here




## Image Generation from Upload

The generate_from_upload function processes an uploaded image and generates variations:

def generate_from_upload(uploaded_image):
    # Implementation details here



## File Structure

• final_model.ipynb: Main script containing all the code

• README.txt: This file

• ffhq.pkl: Pre-trained StyleGAN2 model (to be downloaded)

• fine_tuned_stylegan.pth: Fine-tuned model weights (generated after training)

• full_model_stylegan.pt: Full model save including architecture and weights



## Dependencies

• PyTorch (with CUDA support)

• torchvision

• torchaudio

• ninja

• tqdm

• Gradio

• PIL (Python Imaging Library)

• numpy

## Youtube Link (Video Demo)

Link: https://youtu.be/pkkFaMoNYwY

## Notes

• This project is designed to run on a CUDA-capable GPU for optimal performance.

• The fine-tuning process is computationally intensive and may take several hours or days depending on your dataset size and hardware.

• Hyperparameters in the fine-tuning loop (learning rates, number of epochs, etc.) may need adjustment for your specific use case.

• The project uses a pre-trained FFHQ model as a starting point. You may need to adjust the code if using a different pre-trained model.



## Troubleshooting

• If you encounter CUDA out-of-memory errors, try reducing the batch size or image resolution.

• For NaN losses during training, the code includes checks to skip problematic batches.


## License

This project uses code from the StyleGAN2-ADA-PyTorch repository, which is subject to its own license. Please refer to the original repository for licensing information.

Acknowledgements

This project builds upon the StyleGAN2-ADA-PyTorch implementation by NVIDIA. We acknowledge their contribution to the field of generative models.
