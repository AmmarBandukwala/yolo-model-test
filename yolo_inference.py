import os
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
import cv2
import numpy as np
import argparse
import json
from pathlib import Path
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time
import logging

# Import the model class from training script
# Make sure the SimpleYOLO class is available
class SimpleYOLO(torch.nn.Module):
    """
    Copy of the model architecture from training script
    """
    
    def __init__(self, num_classes=80, num_anchors=3):
        super(SimpleYOLO, self).__init__()
        self.num_classes = num_classes
        self.num_anchors = num_anchors
        
        # Backbone (simplified ResNet-like)
        self.backbone = torch.nn.Sequential(
            # Initial conv
            torch.nn.Conv2d(3, 64, 7, stride=2, padding=3),
            torch.nn.BatchNorm2d(64),
            torch.nn.ReLU(inplace=True),
            torch.nn.MaxPool2d(3, stride=2, padding=1),
            
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
        self.detection_head = torch.nn.Conv2d(
            512, 
            num_anchors * (5 + num_classes),  # 5 = x,y,w,h,confidence
            kernel_size=1
        )
        
    def _make_layer(self, in_channels, out_channels, blocks, stride=1):
        layers = []
        layers.append(torch.nn.Conv2d(in_channels, out_channels, 3, stride, 1))
        layers.append(torch.nn.BatchNorm2d(out_channels))
        layers.append(torch.nn.ReLU(inplace=True))
        
        for _ in range(1, blocks):
            layers.append(torch.nn.Conv2d(out_channels, out_channels, 3, 1, 1))
            layers.append(torch.nn.BatchNorm2d(out_channels))
            layers.append(torch.nn.ReLU(inplace=True))
            
        return torch.nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.backbone(x)
        x = self.detection_head(x)
        return x

class YOLOInference:
    """
    YOLO inference class for object detection
    """
    
    def __init__(self, model_path, num_classes=80, class_names=None, img_size=640, conf_thresh=0.5, nms_thresh=0.4):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.img_size = img_size
        self.conf_thresh = conf_thresh
        self.nms_thresh = nms_thresh
        self.num_classes = num_classes
        
        # Load class names
        if class_names is None:
            self.class_names = [f'class_{i}' for i in range(num_classes)]
        else:
            self.class_names = class_names
            
        # Load model
        self.model = self._load_model(model_path)
        
        # Image preprocessing
        self.transform = transforms.Compose([
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        logging.info(f"Model loaded successfully on {self.device}")
        
    def _load_model(self, model_path):
        """Load the trained model"""
        model = SimpleYOLO(num_classes=self.num_classes)
        
        # Load weights
        if model_path.endswith('.pth'):
            # Direct state dict
            state_dict = torch.load(model_path, map_location=self.device)
            if 'model_state_dict' in state_dict:
                # Checkpoint format
                model.load_state_dict(state_dict['model_state_dict'])
            else:
                # Direct state dict format
                model.load_state_dict(state_dict)
        
        model.to(self.device)
        model.eval()
        return model
    
    def preprocess_image(self, image):
        """Preprocess image for inference"""
        if isinstance(image, str):
            # Load image from path
            image = Image.open(image).convert('RGB')
        elif isinstance(image, np.ndarray):
            # Convert numpy array to PIL
            image = Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        
        # Store original size for scaling back
        orig_size = image.size
        
        # Apply transforms
        image_tensor = self.transform(image).unsqueeze(0)
        
        return image_tensor, orig_size
    
    def postprocess_predictions(self, predictions, orig_size):
        """
        Post-process model predictions to get bounding boxes
        This is a simplified version - real YOLO post-processing is more complex
        """
        # This is where you would implement proper YOLO decoding
        # Including anchor box matching, NMS, etc.
        
        batch_size, channels, height, width = predictions.shape
        num_anchors = 3
        num_attrs = 5 + self.num_classes
        
        # Reshape predictions
        predictions = predictions.view(batch_size, num_anchors, num_attrs, height, width)
        predictions = predictions.permute(0, 1, 3, 4, 2).contiguous()
        
        # Apply sigmoid to confidence and class predictions
        predictions[..., 4] = torch.sigmoid(predictions[..., 4])  # objectness
        predictions[..., 5:] = torch.sigmoid(predictions[..., 5:])  # class probabilities
        
        # Convert to list of detections
        detections = []
        
        # This is a placeholder - implement proper decoding logic
        for i in range(height):
            for j in range(width):
                for a in range(num_anchors):
                    confidence = predictions[0, a, i, j, 4].item()
                    if confidence > self.conf_thresh:
                        # Get class with highest probability
                        class_probs = predictions[0, a, i, j, 5:]
                        class_conf, class_id = torch.max(class_probs, 0)
                        
                        if class_conf.item() > 0.5:  # Additional class confidence threshold
                            # Simplified box coordinates (needs proper implementation)
                            x = (j + predictions[0, a, i, j, 0].item()) / width
                            y = (i + predictions[0, a, i, j, 1].item()) / height
                            w = predictions[0, a, i, j, 2].item()
                            h = predictions[0, a, i, j, 3].item()
                            
                            # Scale back to original image size
                            x_center = x * orig_size[0]
                            y_center = y * orig_size[1]
                            box_w = w * orig_size[0]
                            box_h = h * orig_size[1]
                            
                            # Convert to corner coordinates
                            x1 = x_center - box_w / 2
                            y1 = y_center - box_h / 2
                            x2 = x_center + box_w / 2
                            y2 = y_center + box_h / 2
                            
                            detections.append({
                                'bbox': [x1, y1, x2, y2],
                                'confidence': confidence,
                                'class_id': class_id.item(),
                                'class_name': self.class_names[class_id.item()]
                            })
        
        # Apply NMS (simplified)
        detections = self.apply_nms(detections)
        
        return detections
    
    def apply_nms(self, detections):
        """Apply Non-Maximum Suppression"""
        if not detections:
            return []
        
        # Sort by confidence
        detections.sort(key=lambda x: x['confidence'], reverse=True)
        
        # Group by class
        class_detections = {}
        for det in detections:
            class_id = det['class_id']
            if class_id not in class_detections:
                class_detections[class_id] = []
            class_detections[class_id].append(det)
        
        # Apply NMS per class
        final_detections = []
        for class_id, dets in class_detections.items():
            final_detections.extend(self.nms_per_class(dets))
        
        return final_detections
    
    def nms_per_class(self, detections):
        """NMS for single class"""
        if not detections:
            return []
        
        keep = []
        while detections:
            # Take highest confidence detection
            current = detections.pop(0)
            keep.append(current)
            
            # Remove overlapping detections
            detections = [det for det in detections if self.iou(current['bbox'], det['bbox']) < self.nms_thresh]
        
        return keep
    
    def iou(self, box1, box2):
        """Calculate Intersection over Union"""
        x1_1, y1_1, x2_1, y2_1 = box1
        x1_2, y1_2, x2_2, y2_2 = box2
        
        # Calculate intersection
        x1_i = max(x1_1, x1_2)
        y1_i = max(y1_1, y1_2)
        x2_i = min(x2_1, x2_2)
        y2_i = min(y2_1, y2_2)
        
        if x2_i <= x1_i or y2_i <= y1_i:
            return 0.0
        
        intersection = (x2_i - x1_i) * (y2_i - y1_i)
        
        # Calculate union
        area1 = (x2_1 - x1_1) * (y2_1 - y1_1)
        area2 = (x2_2 - x1_2) * (y2_2 - y1_2)
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0
    
    def detect_image(self, image_path):
        """Run inference on a single image"""
        # Preprocess
        image_tensor, orig_size = self.preprocess_image(image_path)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        start_time = time.time()
        with torch.no_grad():
            predictions = self.model(image_tensor)
        inference_time = time.time() - start_time
        
        # Post-process
        detections = self.postprocess_predictions(predictions, orig_size)
        
        return detections, inference_time
    
    def detect_video(self, video_path, output_path=None):
        """Run inference on video"""
        cap = cv2.VideoCapture(video_path)
        
        if output_path:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            fps = int(cap.get(cv2.CAP_PROP_FPS))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        frame_count = 0
        total_time = 0
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run detection
            detections, inference_time = self.detect_image(frame)
            total_time += inference_time
            frame_count += 1
            
            # Draw detections
            annotated_frame = self.draw_detections(frame, detections)
            
            if output_path:
                out.write(annotated_frame)
            
            # Display (optional)
            cv2.imshow('YOLO Detection', annotated_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cap.release()
        if output_path:
            out.release()
        cv2.destroyAllWindows()
        
        avg_fps = frame_count / total_time if total_time > 0 else 0
        print(f"Processed {frame_count} frames at {avg_fps:.2f} FPS")
    
    def draw_detections(self, image, detections):
        """Draw bounding boxes and labels on image"""
        if isinstance(image, str):
            image = cv2.imread(image)
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            confidence = det['confidence']
            class_name = det['class_name']
            
            # Draw bounding box
            color = self.get_color(det['class_id'])
            cv2.rectangle(image, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name}: {confidence:.2f}"
            label_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 2)[0]
            cv2.rectangle(image, (x1, y1 - label_size[1] - 10), (x1 + label_size[0], y1), color, -1)
            cv2.putText(image, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)
        
        return image
    
    def get_color(self, class_id):
        """Get color for class"""
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), (255, 0, 255),
            (0, 255, 255), (128, 0, 0), (0, 128, 0), (0, 0, 128), (128, 128, 0)
        ]
        return colors[class_id % len(colors)]

def load_class_names(file_path):
    """Load class names from file"""
    if file_path.endswith('.json'):
        with open(file_path, 'r') as f:
            data = json.load(f)
            return data.get('names', [])
    else:
        with open(file_path, 'r') as f:
            return [line.strip() for line in f.readlines()]

def main():
    parser = argparse.ArgumentParser(description='YOLO Inference Script')
    
    # Model parameters
    parser.add_argument('--weights', type=str, required=True,
                        help='Path to trained model weights')
    parser.add_argument('--num-classes', type=int, default=80,
                        help='Number of classes')
    parser.add_argument('--class-names', type=str, default=None,
                        help='Path to class names file')
    
    # Input parameters
    parser.add_argument('--source', type=str, required=True,
                        help='Input source (image/video path or webcam index)')
    parser.add_argument('--output', type=str, default=None,
                        help='Output path for results')
    
    # Inference parameters
    parser.add_argument('--img-size', type=int, default=640,
                        help='Inference image size')
    parser.add_argument('--conf-thresh', type=float, default=0.5,
                        help='Confidence threshold')
    parser.add_argument('--nms-thresh', type=float, default=0.4,
                        help='NMS threshold')
    
    # Display parameters
    parser.add_argument('--save-results', action='store_true',
                        help='Save detection results')
    parser.add_argument('--show', action='store_true',
                        help='Show results')
    
    args = parser.parse_args()
    
    # Load class names
    class_names = None
    if args.class_names:
        class_names = load_class_names(args.class_names)
    
    # Initialize inference
    detector = YOLOInference(
        model_path=args.weights,
        num_classes=args.num_classes,
        class_names=class_names,
        img_size=args.img_size,
        conf_thresh=args.conf_thresh,
        nms_thresh=args.nms_thresh
    )
    
    # Run inference
    if args.source.isdigit():
        # Webcam
        detector.detect_video(int(args.source), args.output)
    elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Video file
        detector.detect_video(args.source, args.output)
    else:
        # Image file
        detections, inference_time = detector.detect_image(args.source)
        
        print(f"Inference time: {inference_time:.3f}s")
        print(f"Detections found: {len(detections)}")
        
        for i, det in enumerate(detections):
            print(f"Detection {i+1}: {det['class_name']} ({det['confidence']:.3f})")
        
        if args.show or args.output:
            # Load original image
            image = cv2.imread(args.source)
            annotated_image = detector.draw_detections(image, detections)
            
            if args.output:
                cv2.imwrite(args.output, annotated_image)
                print(f"Results saved to: {args.output}")
            
            if args.show:
                cv2.imshow('YOLO Detection', annotated_image)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
        
        if args.save_results:
            # Save results to JSON
            results_file = args.source.replace('.jpg', '_results.json').replace('.png', '_results.json')
            with open(results_file, 'w') as f:
                json.dump(detections, f, indent=2)
            print(f"Detection results saved to: {results_file}")

if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    main()