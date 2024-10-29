import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import transforms
from timm import create_model  # Make sure to install timm if you haven't already
import timm
# Assuming your dataload.py is in the same directory or properly imported
from DataLoad import create_default_config, Loader

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Create a config object
config = create_default_config()
config.image_dir = "/home/lucian/University/MSc-Courses/ComputerVision/data/Train/Urban/images_png"
config.mask_dir = "/home/lucian/University/MSc-Courses/ComputerVision/data/Train/Urban/masks_png"
config.batch_size = 4                        # Adjust based on your GPU memory
config.num_workers = 4                       # Number of workers for data loading

# Create a DataLoader instance
train_loader = Loader(config)
import torch
from mmseg.models import build_segmentor
from mmcv import Config

# Load your configuration file for SegFormer
cfg = Config.fromfile('configs/segformer/segformer_b0_512x512.py')  # Path to the config file for SegFormer
cfg.model.decode_head.num_classes = 7  # Adjust number of classes
cfg.model.pretrained = 'https://download.openmmlab.com/mmsegmentation/v0.5/segformer/segformer_b0_512x512_160x160_80k_cityscapes/segformer_b0_512x512_160x160_80k_cityscapes.pth'  # Use a pretrained model if needed

# Build the model
model = build_segmentor(cfg.model)
model = model.to(device)  # Move the model to the appropriate device

# Define loss function and optimizer
criterion = nn.CrossEntropyLoss()  # For multi-class segmentation
optimizer = optim.Adam(model.parameters(), lr=1e-4)

# Training loop for one epoch
model.train()  # Set the model to training mode
for epoch in range(1):  # Loop for one epoch
    running_loss = 0.0
    for images, targets in train_loader:
        images = images.to(device)  # Move images to GPU
        masks = targets["cls"].to(device)  # Move masks to GPU
        
        # Ensure masks have the correct shape for CrossEntropyLoss
        # Masks should be (N, H, W), so we may need to squeeze them if they are (N, 1, H, W)
        if masks.ndim == 4:
            masks = masks.squeeze(1)  # Remove the channel dimension if necessary

        optimizer.zero_grad()  # Zero the gradients
        outputs = model(images)  # Forward pass

        # Ensure outputs are reshaped appropriately for loss computation
        # CrossEntropyLoss expects outputs to be of shape (N, C, H, W)
        # No additional reshaping is typically needed if the model outputs are correct

        print(f'Outputs shape: {outputs.shape}, Masks shape: {masks.shape}')  # Debugging shapes
        
        # Calculate loss
        loss = criterion(outputs, masks)  # Calculate loss
        loss.backward()  # Backward pass
        optimizer.step()  # Optimize the model

        running_loss += loss.item()  # Accumulate loss

    print(f'Epoch [{epoch+1}], Loss: {running_loss / len(train_loader):.4f}')

print('Training complete for one epoch.')
