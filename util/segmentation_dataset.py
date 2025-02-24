import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import numpy as np

class SegmentationDataset(Dataset):
    """
    Dataset for image segmentation tasks with separate image and annotation folders
    """
    def __init__(self, image_dir, transform=None):
        """
        Args:
            image_dir (str): Directory with all input images
            mask_dir (str): Directory with all segmentation masks
            transform (callable, optional): Transform to be applied on input images
            mask_transform (callable, optional): Transform to be applied on masks
        """
        mask_dir = image_dir.replace('images','annotations')
        mask_transform = transform
        self.image_dir = image_dir
        self.mask_dir = mask_dir
        self.transform = transform
        self.mask_transform = mask_transform
        
        # Get all image files
        self.images = sorted([os.path.join(image_dir, f) for f in os.listdir(image_dir) 
                             if f.endswith(('.png', '.jpg', '.jpeg'))])
        
        # Get all mask files - assuming same filename as corresponding image
        self.masks = []
        for img_path in self.images:
            filename = os.path.basename(img_path)
            # Handle different possible file extensions for masks
            mask_name = os.path.splitext(filename)[0] + '.png'  # Common for masks to be PNG
            mask_path = os.path.join(mask_dir, mask_name)
            
            # Check if mask exists with PNG extension, otherwise try other extensions
            if not os.path.exists(mask_path):
                mask_name = os.path.splitext(filename)[0] + '.jpg'
                mask_path = os.path.join(mask_dir, mask_name)
            if not os.path.exists(mask_path):
                mask_name = os.path.splitext(filename)[0] + '.jpeg'
                mask_path = os.path.join(mask_dir, mask_name)
            
            if os.path.exists(mask_path):
                self.masks.append(mask_path)
            else:
                raise FileNotFoundError(f"No mask found for {filename} in {mask_dir}")
        
        # Verify we have the same number of images and masks
        assert len(self.images) == len(self.masks), "Number of images and masks don't match!"

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        # Load image
        img_path = self.images[idx]
        mask_path = self.masks[idx]
        
        image = Image.open(img_path).convert('RGB')
        
        # # Load mask - handling different mask formats
        # mask = Image.open(mask_path)
        
        # # Convert mask to proper format based on your segmentation task
        # # For instance classes, convert to long tensor
        # # For binary segmentation, convert to float tensor
        
        # # Apply transforms
        if self.transform:
            image = self.transform(image)
        
        # if self.mask_transform:
        #     mask = self.mask_transform(mask)
        # else:
        #     # Default conversion for masks - adjust based on your task
        #     mask = torch.from_numpy(np.array(mask)).long()
        
        return image #, mask

# Transforms suitable for segmentation
def get_transforms(img_size=512):
    """
    Create transforms for training segmentation models
    Returns transforms for images and masks
    """
    # Image transforms
    image_transform = transforms.Compose([
        transforms.Resize((img_size, img_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], 
                            std=[0.229, 0.224, 0.225])
    ])
    
    # Mask transforms - typically simpler than image transforms
    # No normalization for masks, just resize and convert to tensor
    mask_transform = transforms.Compose([
        transforms.Resize((img_size, img_size), interpolation=transforms.InterpolationMode.NEAREST),
        transforms.ToTensor()
    ])
    
    return image_transform, mask_transform