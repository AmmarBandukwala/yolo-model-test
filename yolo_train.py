import os
import yaml
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from pathlib import Path
import argparse
import logging
from tqdm import tqdm
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class YOLODataset(torch.utils.data.Dataset):
    """
    Custom dataset for YOLO training
    Expected format: images in 'images/' folder, labels in 'labels/' folder
    Label format: class_id center_x center_y width height (normalized 0-1)
    """
    
    def __init__(self, data_dir, img_size=640, augment=True):
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.augment = augment
        
        # Get all image files
        self.image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            self.image_files.extend(list((self.data_dir / 'images').glob(ext)))
        
        logger.info(f"Found {len(self.image_files)} images in {data_dir}")
        
        # Define transforms
        self.transforms = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self):
        return len(self.image_files)
    
    def __getitem__(self, idx):
        # Load image
        img_path = self.image_files[idx]
        image = Image.open(img_path).convert('RGB')
        
        # Load corresponding label
        label_path = self.data_dir / 'labels' / (img_path.stem + '.txt')
        targets = []
        
        if label_path.exists():
            with open(label_path, 'r') as f:
                for line in f.readlines():
                    if line.strip():
                        class_id, x, y, w, h = map(float, line.strip().split())
                        targets.append([class_id, x, y, w, h])
        
        targets = torch.tensor(targets, dtype=torch.float32)
        
        # Apply transforms
        if self.transforms:
            image = self.transforms(image)
        
        return image, targets

class SimpleYOLO(nn.Module):
    """
    Simplified YOLO architecture for demonstration
    This is a basic implementation - for production use YOLOv5/v8/v10
    """
    
    def __init__(self, num_classes=80, num_anchors=3):
        super(SimpleYOLO, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Backbone (simplified ResNet-like)
        self.backbone = nn.Sequential(
            # Initial conv
            nn.Conv2d(3, 64, 7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(3, stride=2, padding=1),
            
            # Layer 1
            self._make_layer(64, 64, 2),
            # Layer 2  
            self._make_layer(64, 128, 2, stride=2),
            # Layer 3
            self._make_layer(128, 256, 2, stride=2),
            # Layer 4
            self._make_layer(256, 512, 2, stride=2),
        )
        
        # Detection head
        self.detection_head = nn.Conv2d(
            512, 
            num_anchors * (5 + num_classes),  # 5 = x,y,w,h,confidence
            kernel_size=1
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        layers.append(nn.BatchNorm2d(out_channels))
        layers.append(nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            
        return nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.detection_head(x)
        return x

class YOLOLoss(nn.Module):
    """
    Simplified YOLO loss function
    """
    
    def __init__(self, num_classes=80):
        super(YOLOLoss, self).__init__()
        self.num_classes = num_classes
        self.mse_loss = nn.MSELoss()
        self.bce_loss = nn.BCEWithLogitsLoss()
        
    def forward(self, predictions, targets):
        # This is a simplified loss calculation
        # In practice, YOLO loss is much more complex
        batch_size = predictions.size(0)
        
        # Placeholder loss calculation
        # You would implement proper YOLO loss here
        loss = torch.tensor(0.0, requires_grad=True, device=predictions.device)
        
        return loss

def create_data_yaml(data_dir, class_names):
    """Create data.yaml configuration file"""
    data_config = {
        'path': str(data_dir),
        'train': 'images/train',
        'val': 'images/val', 
        'test': 'images/test',
        'nc': len(class_names),
        'names': class_names
    }
    
    yaml_path = Path(data_dir) / 'data.yaml'
    with open(yaml_path, 'w') as f:
        yaml.dump(data_config, f)
    
    return yaml_path

def train_yolo(args):
    """Main training function"""
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logger.info(f"Using device: {device}")
    
    # Create model
    model = SimpleYOLO(num_classes=args.num_classes)
    model.to(device)
    
    # Load pretrained weights if specified
    if args.weights and os.path.exists(args.weights):
        logger.info(f"Loading weights from {args.weights}")
        model.load_state_dict(torch.load(args.weights, map_location=device))
    
    # Create datasets
    train_dataset = YOLODataset(
        args.data_dir + '/train', 
        img_size=args.img_size,
        augment=True
    )
    
    val_dataset = YOLODataset(
        args.data_dir + '/val',
        img_size=args.img_size, 
        augment=False
    )
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.workers,
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.workers,
        collate_fn=collate_fn
    )
    
    # Loss function and optimizer
    criterion = YOLOLoss(num_classes=args.num_classes)
    optimizer = optim.AdamW(
        model.parameters(),
        lr=args.learning_rate,
        weight_decay=args.weight_decay
    )
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.StepLR(
        optimizer, 
        step_size=args.step_size,
        gamma=args.gamma
    )
    
    # Training loop
    best_loss = float('inf')
    
    for epoch in range(args.epochs):
        model.train()
        train_loss = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{args.epochs}')
        
        for batch_idx, (images, targets) in enumerate(pbar):
            images = images.to(device)
            
            # Forward pass
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, targets)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            train_loss += loss.item()
            
            # Update progress bar
            pbar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Avg Loss': f'{train_loss/(batch_idx+1):.4f}'
            })
        
        # Validation
        model.eval()
        val_loss = 0.0
        
        with torch.no_grad():
            for images, targets in val_loader:
                images = images.to(device)
                outputs = model(images)
                loss = criterion(outputs, targets)
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        
        logger.info(f'Epoch {epoch+1}: Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}')
        
        # Save best model
        if val_loss < best_loss:
            best_loss = val_loss
            torch.save(model.state_dict(), 'best_model.pth')
            logger.info(f'New best model saved with validation loss: {val_loss:.4f}')
        
        # Save checkpoint
        if (epoch + 1) % args.save_freq == 0:
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': val_loss,
            }, f'checkpoint_epoch_{epoch+1}.pth')
        
        scheduler.step()

def collate_fn(batch):
    """Custom collate function for YOLO dataset"""
    images, targets = zip(*batch)
    images = torch.stack(images, 0)
    return images, targets

def main():
    parser = argparse.ArgumentParser(description='YOLO Training Script')
    
    # Data parameters
    parser.add_argument('--data-dir', type=str, required=True,
                        help='Path to dataset directory')
    parser.add_argument('--num-classes', type=int, default=80,
                        help='Number of classes')
    parser.add_argument('--img-size', type=int, default=640,
                        help='Image size for training')
    
    # Training parameters
    parser.add_argument('--epochs', type=int, default=100,
                        help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                        help='Batch size for training')
    parser.add_argument('--learning-rate', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--weight-decay', type=float, default=0.0005,
                        help='Weight decay')
    
    # Scheduler parameters
    parser.add_argument('--step-size', type=int, default=30,
                        help='Step size for learning rate scheduler')
    parser.add_argument('--gamma', type=float, default=0.1,
                        help='Gamma for learning rate scheduler')
    
    # Other parameters
    parser.add_argument('--workers', type=int, default=4,
                        help='Number of data loading workers')
    parser.add_argument('--weights', type=str, default='',
                        help='Path to pretrained weights')
    parser.add_argument('--save-freq', type=int, default=10,
                        help='Save checkpoint every N epochs')
    
    args = parser.parse_args()
    
    # Start training
    logger.info("Starting YOLO training...")
    train_yolo(args)
    logger.info("Training completed!")

if __name__ == '__main__':
    main()