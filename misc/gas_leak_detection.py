import cv2
import numpy as np
import matplotlib.pyplot as plt
import argparse
import time
from collections import deque
import logging
import json
from pathlib import Path

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class GasLeakDetector:
    """
    Gas Leak Detection using Computer Vision Techniques
    
    Methods supported:
    1. Thermal distortion detection (heat shimmer)
    2. Background subtraction (visible plumes)
    3. Optical flow analysis (air movement)
    4. Edge distortion detection
    5. Histogram analysis (opacity changes)
    """
    
    def __init__(self, method='thermal', sensitivity=0.5, history_size=30):
        self.method = method
        self.sensitivity = sensitivity
        self.history_size = history_size
        
        # Initialize detection components
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=500, varThreshold=16, detectShadows=True
        )
        
        # Frame history for analysis
        self.frame_history = deque(maxlen=history_size)
        self.baseline_established = False
        self.baseline_frame = None
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(15, 15),
            maxLevel=2,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03)
        )
        
        # Feature detection parameters
        self.feature_params = dict(
            maxCorners=100,
            qualityLevel=0.3,
            minDistance=7,
            blockSize=7
        )
        
        # Results storage
        self.detection_results = []
        
    def preprocess_frame(self, frame):
        """Preprocess frame for gas detection"""
        # Convert to grayscale
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Apply Gaussian blur to reduce noise
        blurred = cv2.GaussianBlur(gray, (5, 5), 0)
        
        # Optional: histogram equalization
        equalized = cv2.equalizeHist(blurred)
        
        return gray, blurred, equalized
    
    def detect_thermal_distortion(self, frame, prev_frame):
        """
        Detect thermal distortion/heat shimmer
        Uses optical flow to identify unusual air movement patterns
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = cv2.cvtColor(prev_frame, cv2.COLOR_BGR2GRAY)
        
        # Detect feature points
        corners = cv2.goodFeaturesToTrack(prev_gray, **self.feature_params)
        
        if corners is not None:
            # Calculate optical flow
            next_pts, status, error = cv2.calcOpticalFlowPyrLK(
                prev_gray, gray, corners, None, **self.lk_params
            )
            
            # Select good points
            good_new = next_pts[status == 1]
            good_old = corners[status == 1]
            
            # Calculate flow magnitude and direction
            flow_vectors = good_new - good_old
            flow_magnitude = np.sqrt(flow_vectors[:, 0]**2 + flow_vectors[:, 1]**2)
            
            # Detect anomalous flow patterns (potential thermal distortion)
            anomaly_threshold = np.mean(flow_magnitude) + 2 * np.std(flow_magnitude)
            anomalous_points = good_new[flow_magnitude > anomaly_threshold]
            
            # Create heat map of distortion
            distortion_mask = np.zeros(gray.shape, dtype=np.uint8)
            if len(anomalous_points) > 0:
                for point in anomalous_points:
                    cv2.circle(distortion_mask, tuple(point.astype(int)), 10, 255, -1)
                
                # Apply Gaussian blur to create smooth distortion regions
                distortion_mask = cv2.GaussianBlur(distortion_mask, (21, 21), 0)
            
            return distortion_mask, anomalous_points, flow_magnitude
        
        return np.zeros(gray.shape, dtype=np.uint8), [], []
    
    def detect_background_changes(self, frame):
        """
        Detect visible gas plumes using background subtraction
        """
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows
        fg_mask[fg_mask == 127] = 0
        
        # Morphological operations to clean up the mask
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Find contours
        contours, _ = cv2.findContours(fg_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        # Filter contours by area and aspect ratio
        gas_regions = []
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > 100:  # Minimum area threshold
                x, y, w, h = cv2.boundingRect(contour)
                aspect_ratio = w / h
                
                # Gas plumes often have specific characteristics
                if 0.5 < aspect_ratio < 3.0:  # Not too elongated
                    gas_regions.append({
                        'contour': contour,
                        'area': area,
                        'bbox': (x, y, w, h),
                        'aspect_ratio': aspect_ratio
                    })
        
        return fg_mask, gas_regions
    
    def detect_edge_distortion(self, frame, baseline):
        """
        Detect edge distortion that might indicate gas presence
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
        
        # Edge detection on current frame and baseline
        edges_current = cv2.Canny(gray, 50, 150)
        edges_baseline = cv2.Canny(baseline_gray, 50, 150)
        
        # Find difference in edge patterns
        edge_diff = cv2.absdiff(edges_current, edges_baseline)
        
        # Apply morphological operations
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        edge_diff = cv2.morphologyEx(edge_diff, cv2.MORPH_CLOSE, kernel)
        
        # Threshold to find significant changes
        _, distortion_mask = cv2.threshold(edge_diff, 30, 255, cv2.THRESH_BINARY)
        
        return distortion_mask
    
    def analyze_histogram_changes(self, frame, baseline):
        """
        Analyze histogram changes that might indicate gas opacity
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        baseline_gray = cv2.cvtColor(baseline, cv2.COLOR_BGR2GRAY)
        
        # Calculate histograms
        hist_current = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist_baseline = cv2.calcHist([baseline_gray], [0], None, [256], [0, 256])
        
        # Calculate histogram correlation
        correlation = cv2.compareHist(hist_current, hist_baseline, cv2.HISTCMP_CORREL)
        
        # Calculate chi-square distance
        chi_square = cv2.compareHist(hist_current, hist_baseline, cv2.HISTCMP_CHISQR)
        
        # Detect significant changes
        correlation_threshold = 0.95  # High correlation = similar images
        chi_square_threshold = 1000   # Low chi-square = similar histograms
        
        opacity_detected = correlation < correlation_threshold or chi_square > chi_square_threshold
        
        return opacity_detected, correlation, chi_square
    
    def quantify_gas_concentration(self, detection_mask, method_confidence):
        """
        Attempt to quantify gas concentration (very approximate)
        Note: This is highly unreliable without calibration
        """
        # Calculate affected area
        total_pixels = detection_mask.shape[0] * detection_mask.shape[1]
        affected_pixels = np.count_nonzero(detection_mask)
        affected_percentage = (affected_pixels / total_pixels) * 100
        
        # Calculate intensity of detection
        mean_intensity = np.mean(detection_mask[detection_mask > 0]) if affected_pixels > 0 else 0
        normalized_intensity = mean_intensity / 255.0
        
        # Rough concentration estimate (arbitrary units)
        # This would need extensive calibration with real gas concentrations
        estimated_concentration = affected_percentage * normalized_intensity * method_confidence
        
        return {
            'affected_area_percent': affected_percentage,
            'mean_intensity': normalized_intensity,
            'estimated_concentration_au': estimated_concentration,  # Arbitrary units
            'confidence_score': method_confidence,
            'affected_pixels': affected_pixels
        }
    
    def detect_gas_leak(self, frame, prev_frame=None):
        """
        Main detection function using selected method
        """
        detection_mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        method_confidence = 0.0
        detection_info = {}
        
        # Store frame in history
        self.frame_history.append(frame.copy())
        
        # Establish baseline if needed
        if not self.baseline_established and len(self.frame_history) >= 10:
            self.baseline_frame = self.frame_history[0].copy()
            self.baseline_established = True
            logger.info("Baseline frame established")
        
        if self.method == 'thermal' and prev_frame is not None:
            # Thermal distortion detection
            distortion_mask, anomalous_points, flow_magnitudes = self.detect_thermal_distortion(frame, prev_frame)
            detection_mask = distortion_mask
            method_confidence = min(len(anomalous_points) / 20.0, 1.0)  # Normalize to 0-1
            detection_info = {
                'anomalous_points': len(anomalous_points),
                'max_flow_magnitude': np.max(flow_magnitudes) if len(flow_magnitudes) > 0 else 0
            }
            
        elif self.method == 'background':
            # Background subtraction
            fg_mask, gas_regions = self.detect_background_changes(frame)
            detection_mask = fg_mask
            total_area = sum([region['area'] for region in gas_regions])
            method_confidence = min(total_area / 10000.0, 1.0)  # Normalize
            detection_info = {
                'gas_regions': len(gas_regions),
                'total_affected_area': total_area
            }
            
        elif self.method == 'edge' and self.baseline_established:
            # Edge distortion detection
            detection_mask = self.detect_edge_distortion(frame, self.baseline_frame)
            edge_pixels = np.count_nonzero(detection_mask)
            method_confidence = min(edge_pixels / 5000.0, 1.0)  # Normalize
            detection_info = {
                'edge_distortion_pixels': edge_pixels
            }
            
        elif self.method == 'histogram' and self.baseline_established:
            # Histogram analysis
            opacity_detected, correlation, chi_square = self.analyze_histogram_changes(frame, self.baseline_frame)
            method_confidence = (1 - correlation) if opacity_detected else 0.0
            detection_info = {
                'opacity_detected': opacity_detected,
                'histogram_correlation': correlation,
                'chi_square_distance': chi_square
            }
            
            # Create a simple mask for visualization
            if opacity_detected:
                detection_mask = np.ones(frame.shape[:2], dtype=np.uint8) * 128
        
        # Quantify concentration (approximate)
        concentration_data = self.quantify_gas_concentration(detection_mask, method_confidence)
        
        # Store results
        result = {
            'timestamp': time.time(),
            'method': self.method,
            'detection_confidence': method_confidence,
            'concentration_data': concentration_data,
            'method_specific_info': detection_info
        }
        self.detection_results.append(result)
        
        return detection_mask, method_confidence, concentration_data
    
    def visualize_detection(self, frame, detection_mask, confidence, concentration_data):
        """
        Create visualization of gas detection results
        """
        result_frame = frame.copy()
        
        # Overlay detection mask
        if np.any(detection_mask > 0):
            # Create colored overlay
            overlay = np.zeros_like(frame)
            overlay[:, :, 2] = detection_mask  # Red channel
            overlay[:, :, 1] = detection_mask // 2  # Green channel (for orange effect)
            
            # Blend with original frame
            alpha = 0.3
            result_frame = cv2.addWeighted(result_frame, 1 - alpha, overlay, alpha, 0)
        
        # Add text information
        info_text = [
            f"Method: {self.method}",
            f"Confidence: {confidence:.2f}",
            f"Affected Area: {concentration_data['affected_area_percent']:.2f}%",
            f"Est. Concentration: {concentration_data['estimated_concentration_au']:.2f} AU"
        ]
        
        y_offset = 30
        for i, text in enumerate(info_text):
            cv2.putText(result_frame, text, (10, y_offset + i * 25), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Add warning if high concentration detected
        if concentration_data['estimated_concentration_au'] > 5.0:
            cv2.putText(result_frame, "POTENTIAL GAS LEAK DETECTED", (10, result_frame.shape[0] - 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
        
        return result_frame
    
    def save_results(self, output_path):
        """Save detection results to JSON file"""
        results_dict = {
            'method': self.method,
            'total_detections': len(self.detection_results),
            'results': self.detection_results
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_dict, f, indent=2)
        
        logger.info(f"Results saved to {output_path}")

def process_video_file(video_path, detector, output_path=None, show_display=True):
    """Process video file for gas leak detection"""
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Error opening video file: {video_path}")
        return
    
    # Video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    logger.info(f"Processing video: {width}x{height}, {fps} FPS, {total_frames} frames")
    
    # Setup video writer if output specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    frame_count = 0
    prev_frame = None
    detection_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        
        # Run detection
        detection_mask, confidence, concentration_data = detector.detect_gas_leak(frame, prev_frame)
        
        if confidence > 0.3:  # Detection threshold
            detection_count += 1
        
        # Create visualization
        result_frame = detector.visualize_detection(frame, detection_mask, confidence, concentration_data)
        
        # Add frame counter
        cv2.putText(result_frame, f"Frame: {frame_count}/{total_frames}", 
                   (width - 200, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        # Save to output video
        if writer:
            writer.write(result_frame)
        
        # Display
        if show_display:
            cv2.imshow('Gas Leak Detection', result_frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        prev_frame = frame.copy()
        
        # Progress update
        if frame_count % 30 == 0:
            progress = (frame_count / total_frames) * 100
            logger.info(f"Progress: {progress:.1f}%, Detections: {detection_count}")
    
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    logger.info(f"Processing complete. Total detections: {detection_count}/{frame_count} frames")

def process_webcam(detector, camera_index=0):
    """Process live webcam feed"""
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        logger.error(f"Error opening camera {camera_index}")
        return
    
    logger.info("Starting webcam processing. Press 'q' to quit, 's' to save current frame")
    
    frame_count = 0
    prev_frame = None
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.error("Error reading from camera")
            break
        
        frame_count += 1
        
        # Run detection
        detection_mask, confidence, concentration_data = detector.detect_gas_leak(frame, prev_frame)
        
        # Create visualization
        result_frame = detector.visualize_detection(frame, detection_mask, confidence, concentration_data)
        
        # Add FPS counter
        cv2.putText(result_frame, f"Frame: {frame_count}", (10, result_frame.shape[0] - 10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        # Display
        cv2.imshow('Gas Leak Detection - Live', result_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('s'):
            # Save current frame
            timestamp = int(time.time())
            cv2.imwrite(f'gas_detection_frame_{timestamp}.jpg', result_frame)
            logger.info(f"Frame saved as gas_detection_frame_{timestamp}.jpg")
        
        prev_frame = frame.copy()
    
    cap.release()
    cv2.destroyAllWindows()

def main():
    parser = argparse.ArgumentParser(description='Gas Leak Detection using OpenCV')
    
    parser.add_argument('--source', type=str, default='0',
                       help='Input source: video file path, image file, or camera index (default: 0)')
    parser.add_argument('--method', type=str, default='thermal',
                       choices=['thermal', 'background', 'edge', 'histogram'],
                       help='Detection method')
    parser.add_argument('--sensitivity', type=float, default=0.5,
                       help='Detection sensitivity (0.0-1.0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Output video path')
    parser.add_argument('--save-results', type=str, default=None,
                       help='Save detection results to JSON file')
    parser.add_argument('--no-display', action='store_true',
                       help='Disable video display')
    
    args = parser.parse_args()
    
    # Initialize detector
    detector = GasLeakDetector(
        method=args.method,
        sensitivity=args.sensitivity
    )
    
    logger.info(f"Starting gas leak detection with method: {args.method}")
    logger.info("WARNING: This is experimental and should not be used for safety-critical applications!")
    
    # Process input
    if args.source.isdigit():
        # Webcam
        process_webcam(detector, int(args.source))
    elif args.source.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
        # Video file
        process_video_file(args.source, detector, args.output, not args.no_display)
    else:
        # Single image
        image = cv2.imread(args.source)
        if image is None:
            logger.error(f"Error loading image: {args.source}")
            return
        
        detection_mask, confidence, concentration_data = detector.detect_gas_leak(image)
        result_image = detector.visualize_detection(image, detection_mask, confidence, concentration_data)
        
        logger.info(f"Detection confidence: {confidence:.2f}")
        logger.info(f"Estimated concentration: {concentration_data['estimated_concentration_au']:.2f} AU")
        
        if args.output:
            cv2.imwrite(args.output, result_image)
            logger.info(f"Result saved to {args.output}")
        
        if not args.no_display:
            cv2.imshow('Gas Leak Detection', result_image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
    
    # Save results if requested
    if args.save_results:
        detector.save_results(args.save_results)

if __name__ == '__main__':
    main()