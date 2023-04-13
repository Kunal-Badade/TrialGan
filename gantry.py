import streamlit as st
import torch
use_gpu = True if torch.cuda.is_available() else False

# trained on high-quality celebrity faces "celebA" dataset
# this model outputs 512 x 512 pixel images
model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
                       'PGAN', model_name='celebAHQ-512',
                       pretrained=True, useGPU=use_gpu)
# this model outputs 256 x 256 pixel images
# model = torch.hub.load('facebookresearch/pytorch_GAN_zoo:hub',
#                        'PGAN', model_name='celebAHQ-256',
#                        pretrained=True, useGPU=use_gpu)

num_images = 4
noise, _ = model.buildNoiseData(num_images)
with torch.no_grad():
    generated_images = model.test(noise)

# let's plot these images using torchvision and matplotlib
import matplotlib.pyplot as plt
import torchvision
grid = torchvision.utils.make_grid(generated_images.clamp(min=-1, max=1), scale_each=True, normalize=True)

# create a new figure
fig, ax = plt.subplots()

# display the image on the new figure
ax.imshow(grid.permute(1, 2, 0).cpu().numpy())

# hide the axes and labels
ax.axis('off')
plt.subplots_adjust(wspace=0, hspace=0)

# convert the Matplotlib figure to a PIL Image
import io
from PIL import Image
buf = io.BytesIO()
fig.savefig(buf, format='png')
buf.seek(0)
pil_image = Image.open(buf)

# show the image in Streamlit
st.image(pil_image, caption="Generated Images", use_column_width=True)
