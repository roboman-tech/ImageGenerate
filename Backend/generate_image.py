from diffusers import StableDiffusionXLPipeline
import torch
import matplotlib.pyplot as plt

device = "cuda" if torch.cuda.is_available() else "cpu"

print(torch.__version__)  # Should show PyTorch version
print(torch.cuda.is_available())  # Should return True if CUDA is available
print(torch.version.cuda)  # Should show the installed CUDA version (e.g., '12.1')
print(torch.cuda.get_device_name(0))  # Should print the name of the GPU (e.g., RTX 4070)
print(f"Using device: {device}")


model_path = "D:/Source/models/sdxl-base-1.0"  # folder containing sd_xl_base_1.0.safetensors

# Load the model (VAE is included)
pipe = StableDiffusionXLPipeline.from_pretrained(
    model_path,
    torch_dtype=torch.float16
).to(device)

prompt = "freelancer children playing in the park, sunny day, vibrant colors, high detail, 4k resolution"
image = pipe(prompt=prompt, num_inference_steps=70, guidance_scale=7.5).images[0]

plt.imshow(image)
plt.axis("off")
plt.show()
image.save("D:/Source/ImageGeneration/generated_image_xl.png")
print("Saved as generated_image_xl.png")
