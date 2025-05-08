#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Facial Region Temporal Flicker Feature Extraction for Deepfake Detection
------------------------------------------------------------------------
This script analyzes temporal flicker in face regions of videos by:
1. Using InsightFace to detect and extract face areas
2. Measuring pixel intensity variance across frames in face regions only
3. Computing temporal flicker features for deepfake detection
4. Visualizing results with t-SNE and UMAP

Uses CUDA for accelerated processing through PyTorch.
"""

import os
import cv2
import numpy as np
import torch
import random
import argparse
import pandas as pd
from tqdm import tqdm
from pathlib import Path
import matplotlib.pyplot as plt
import time
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler
import warnings

# Import UMAP with error handling
try:
    import umap
    UMAP_AVAILABLE = True
except ImportError:
    UMAP_AVAILABLE = False
    warnings.warn("UMAP not installed. UMAP visualization will be skipped. Install with 'pip install umap-learn'")

# Import InsightFace with error handling
try:
    import insightface
    from insightface.app import FaceAnalysis
    INSIGHTFACE_AVAILABLE = True
except ImportError:
    INSIGHTFACE_AVAILABLE = False
    warnings.warn("InsightFace not installed. Install with 'pip install insightface'")

# Set random seeds for reproducibility
def set_seed(seed=42):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

# Set the seed
set_seed(42)

# Check if CUDA is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using PyTorch device: {device}")

# Optimize CUDA performance if available
if torch.cuda.is_available():
    # Set the current device
    torch.cuda.set_device(0)
    
    # Print device properties
    cuda_device = torch.cuda.current_device()
    print(f"Using CUDA Device: {torch.cuda.get_device_name(cuda_device)}")
    print(f"Device Capability: {torch.cuda.get_device_capability(cuda_device)}")
    
    # Print memory stats
    print(f"Memory allocated: {torch.cuda.memory_allocated(cuda_device) / 1024**2:.2f} MB")
    print(f"Memory cached: {torch.cuda.memory_reserved(cuda_device) / 1024**2:.2f} MB")
    
    # Optimize settings for better performance
    torch.backends.cudnn.benchmark = True

def initialize_face_detector(use_cuda=False):
    """
    Initialize the InsightFace face detector (CPU only)
    
    Args:
        use_cuda (bool): Ignored, always uses CPU for face detection
    
    Returns:
        FaceAnalysis object: Initialized face detector
    """
    if not INSIGHTFACE_AVAILABLE:
        raise ImportError("InsightFace is required but not installed. Install with 'pip install insightface'")
    
    # Initialize face analyzer - force CPU provider regardless of use_cuda parameter
    face_app = FaceAnalysis(
        name="buffalo_l",  # Use the large model for better accuracy
        providers=['CPUExecutionProvider']  # Always use CPU for InsightFace
    )
    face_app.prepare(ctx_id=-1, det_size=(640, 640))  # ctx_id=-1 forces CPU
    
    print("InsightFace initialized with CPU provider")
    return face_app

def extract_face_regions(frame, face_app, padding=0.2):
    """
    Extract face regions from a frame
    
    Args:
        frame (np.array): Input frame
        face_app (FaceAnalysis): InsightFace face detector
        padding (float): Padding to add around the face (percentage of face size)
    
    Returns:
        list: List of face region images and bounding boxes
    """
    if frame is None:
        return []
    
    # Detect faces
    faces = face_app.get(frame)
    face_regions = []
    
    for face in faces:
        # Get bounding box
        bbox = face.bbox.astype(np.int32)
        x1, y1, x2, y2 = bbox
        
        # Add padding
        height, width = y2 - y1, x2 - x1
        pad_h, pad_w = int(height * padding), int(width * padding)
        
        # Ensure coordinates are within image boundaries
        x1_padded = max(0, x1 - pad_w)
        y1_padded = max(0, y1 - pad_h)
        x2_padded = min(frame.shape[1], x2 + pad_w)
        y2_padded = min(frame.shape[0], y2 + pad_h)
        
        # Extract face region
        face_region = frame[y1_padded:y2_padded, x1_padded:x2_padded]
        
        # Only add if face region is valid
        if face_region.size > 0:
            face_regions.append({
                'face_region': face_region,
                'bbox': (x1_padded, y1_padded, x2_padded, y2_padded)
            })
    
    return face_regions

def extract_facial_flicker_features(video_path, face_app, sample_frames=None, use_cuda=True):
    """
    Extract temporal flicker features from face regions in a video using PyTorch CUDA acceleration
    
    Args:
        video_path (str): Path to the video file
        face_app (FaceAnalysis): InsightFace face detector
        sample_frames (int, optional): Number of frames to sample. If None, use all frames.
        use_cuda (bool): Whether to use CUDA for processing
        
    Returns:
        dict: Dictionary of temporal flicker features
    """
    # Using PyTorch for CUDA acceleration when available
    if torch.cuda.is_available() and use_cuda:
        print(f"Using PyTorch with CUDA acceleration")
    else:
        print(f"Using CPU for video processing")
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return None
    
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    if total_frames <= 1:
        print(f"Video has insufficient frames: {video_path}")
        return None
    
    # Sample frames or use all frames
    if sample_frames and sample_frames < total_frames:
        frame_indices = sorted(random.sample(range(total_frames), min(sample_frames, total_frames)))
    else:
        frame_indices = list(range(total_frames))
    
    # Read first frame to detect face and determine tracking area
    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[0])
    ret, first_frame = cap.read()
    
    if not ret or first_frame is None:
        print(f"Error reading first frame from video: {video_path}")
        return None
    
    # Detect faces in first frame
    face_regions = extract_face_regions(first_frame, face_app)
    
    if not face_regions:
        print(f"No faces detected in video: {video_path}")
        # Try a few more frames before giving up
        for i in range(min(10, len(frame_indices))):
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_indices[i])
            ret, frame = cap.read()
            if ret and frame is not None:
                face_regions = extract_face_regions(frame, face_app)
                if face_regions:
                    break
        
        if not face_regions:
            return None
    
    # Use the largest face if multiple faces detected
    if len(face_regions) > 1:
        # Find the largest face by area
        largest_face_idx = 0
        largest_area = 0
        for i, face_data in enumerate(face_regions):
            x1, y1, x2, y2 = face_data['bbox']
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_face_idx = i
        
        primary_face = face_regions[largest_face_idx]
    else:
        primary_face = face_regions[0]
    
    # Store bounding box of primary face
    primary_bbox = primary_face['bbox']
    
    # Store face region intensities for each frame
    face_intensities = []
    temporal_diff_maps = []
    prev_face_gray = None
    
    # Read frames and extract face regions
    for i in tqdm(frame_indices, desc="Extracting face flicker"):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        
        if not ret or frame is None:
            continue
        
        # Extract face region using the primary face bounding box
        x1, y1, x2, y2 = primary_bbox
        face_region = frame[y1:y2, x1:x2]
        
        if face_region.size == 0:
            continue
        
        # Convert to grayscale
        face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
        
        # Calculate intensity statistics for face region
        if torch.cuda.is_available() and use_cuda:
            # Use PyTorch CUDA for acceleration
            face_tensor = torch.from_numpy(face_gray).cuda().float()
            mean_intensity = torch.mean(face_tensor).item()
            
            if prev_face_gray is not None:
                prev_face_tensor = torch.from_numpy(prev_face_gray).cuda().float()
                diff_map = torch.abs(face_tensor - prev_face_tensor)
                temporal_diff_maps.append(diff_map.cpu().numpy())
            
            # Free up CUDA memory
            del face_tensor
            if 'prev_face_tensor' in locals():
                del prev_face_tensor
            torch.cuda.empty_cache()
            
        else:
            mean_intensity = np.mean(face_gray)
            
            if prev_face_gray is not None:
                diff_map = np.abs(face_gray.astype(np.float32) - prev_face_gray.astype(np.float32))
                temporal_diff_maps.append(diff_map)
        
        face_intensities.append(mean_intensity)
        prev_face_gray = face_gray
    
    cap.release()
    
    # Make sure CUDA memory is properly cleared
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    
    if len(face_intensities) <= 1:
        print(f"Could not extract sufficient face frames: {video_path}")
        return None
    
    # Convert to numpy array for calculations
    face_intensities = np.array(face_intensities)
    
    # Calculate temporal differences
    temporal_diffs = np.diff(face_intensities)
    
    # Calculate features
    features = {
        'video_path': video_path,
        'video_length': total_frames / fps,  # Video length in seconds
        'fps': fps,
        'total_frames': total_frames,
        'face_mean_intensity': np.mean(face_intensities),
        'face_std_intensity': np.std(face_intensities),
        'face_temporal_variance': np.var(temporal_diffs),
        'face_temporal_std': np.std(temporal_diffs),
        'face_max_intensity_diff': np.max(np.abs(temporal_diffs)),
        'face_mean_abs_diff': np.mean(np.abs(temporal_diffs)),
        'face_median_abs_diff': np.median(np.abs(temporal_diffs)),
        'face_intensity_range': np.max(face_intensities) - np.min(face_intensities),
        # Normalized features to account for different video lengths
        'face_flicker_density': np.sum(np.abs(temporal_diffs) > np.std(temporal_diffs)) / len(temporal_diffs),
        'face_normalized_variance': np.var(temporal_diffs) / (np.mean(face_intensities) + 1e-8),
    }
    
    # Calculate spatial flicker features if we have diff maps
    if temporal_diff_maps:
        # Convert to numpy arrays if they're PyTorch tensors
        diff_maps_np = [diff_map if isinstance(diff_map, np.ndarray) else diff_map.cpu().numpy() 
                        for diff_map in temporal_diff_maps]
        
        # Calculate mean spatial variance of differences
        spatial_variances = [np.var(diff_map) for diff_map in diff_maps_np]
        features['face_spatial_flicker_mean'] = np.mean(spatial_variances)
        features['face_spatial_flicker_std'] = np.std(spatial_variances)
        
        # Calculate flicker concentration (how localized the flicker is)
        # Higher values mean more localized flickering, which may indicate deepfakes
        flicker_concentrations = []
        for diff_map in diff_maps_np:
            # Threshold the diff map to focus on significant changes
            threshold = np.mean(diff_map) + np.std(diff_map)
            significant_pixels = diff_map > threshold
            if np.sum(significant_pixels) > 0:
                # Calculate how clustered the significant pixels are
                # by comparing to a uniform distribution
                expected_ratio = np.sum(significant_pixels) / diff_map.size
                rows, cols = np.where(significant_pixels)
                if len(rows) > 1:
                    # Calculate standard deviation of positions as a measure of clustering
                    position_std = np.std(rows) + np.std(cols)
                    # Normalize by image dimensions
                    position_std_normalized = position_std / (diff_map.shape[0] + diff_map.shape[1])
                    flicker_concentrations.append(1.0 - position_std_normalized)
        
        if flicker_concentrations:
            features['face_flicker_concentration'] = np.mean(flicker_concentrations)
    
    # Calculate frequency domain features using FFT
    if len(face_intensities) > 10:  # Ensure we have enough samples for FFT
        fft_values = np.abs(np.fft.rfft(face_intensities - np.mean(face_intensities)))
        # Normalize by number of frames
        fft_values = fft_values / len(face_intensities)
        
        # Calculate energy in different frequency bands
        if len(fft_values) >= 4:
            freq_bands = np.array_split(fft_values[1:], 3)  # Skip DC component
            features.update({
                'face_low_freq_energy': np.sum(freq_bands[0]**2),
                'face_mid_freq_energy': np.sum(freq_bands[1]**2),
                'face_high_freq_energy': np.sum(freq_bands[2]**2),
                'face_fft_peak_magnitude': np.max(fft_values[1:]),  # Skip DC component
                'face_fft_peak_frequency': np.argmax(fft_values[1:]) + 1,  # Skip DC component
            })
    
    return features

def process_dataset(root_folder, output_file, face_app, sample_frames=None, use_cuda=True, 
                   max_train_videos=100, max_test_videos=20):
    """
    Process all videos in the dataset and save features to CSV.
    
    Args:
        root_folder (str): Path to the root folder containing categories
        output_file (str): Path to save the output CSV file
        face_app (FaceAnalysis): InsightFace face detector
        sample_frames (int, optional): Number of frames to sample per video
        use_cuda (bool): Whether to use CUDA for processing
        max_train_videos (int): Maximum number of training videos to process (per label)
        max_test_videos (int): Maximum number of testing videos to process (per label)
    """
    root_path = Path(root_folder)
    results = []
    
    # Check if the root folder exists
    if not root_path.exists():
        print(f"ERROR: Dataset root folder '{root_folder}' does not exist.")
        print(f"Current working directory: {os.getcwd()}")
        print("Please provide a valid dataset path with --root argument.")
        return None
    
    # Check CUDA capability
    cuda_available = torch.cuda.is_available() and use_cuda
    
    if cuda_available:
        print(f"CUDA is available: {torch.cuda.get_device_name(0)}")
        print(f"Memory allocated: {torch.cuda.memory_allocated(0) / 1024**2:.2f} MB")
        print(f"Memory reserved: {torch.cuda.memory_reserved(0) / 1024**2:.2f} MB")
    
    # Find all subdirectories in the root folder
    if root_path.is_dir():
        categories = [d for d in root_path.iterdir() if d.is_dir()]
        
        if not categories:
            print(f"WARNING: No subdirectories found in {root_folder}")
            print("Looking for video files directly in the root folder...")
            
            # Try to find videos directly in the root folder
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MOV']
            video_files = []
            for ext in video_extensions:
                video_files.extend(list(root_path.glob(f'*{ext}')))
            
            if video_files:
                print(f"Found {len(video_files)} videos in the root folder.")
                # Process videos directly from root folder (assuming all are "real" for now)
                try:
                    for video_file in tqdm(video_files, desc="Processing videos"):
                        with torch.no_grad():
                            features = extract_facial_flicker_features(
                                str(video_file),
                                face_app,
                                sample_frames,
                                use_cuda=use_cuda
                            )
                        
                        if features:
                            features['category'] = 'unknown'
                            features['label'] = 'unknown'  # No label info available
                            results.append(features)
                            
                            # Periodically save results and flush GPU memory
                            if len(results) % 5 == 0 and cuda_available:
                                torch.cuda.empty_cache()
                except Exception as e:
                    print(f"Error processing videos: {e}")
            else:
                print(f"No video files found in {root_folder}")
                return None
    else:
        print(f"ERROR: {root_folder} is not a directory")
        return None
    
    for category in categories:
        # Determine max videos based on category
        if category.name.lower() == 'train':
            max_videos = max_train_videos
        elif category.name.lower() == 'test':
            max_videos = max_test_videos
        else:
            max_videos = max_train_videos  # Default to training limit
            
        print(f"Processing category: {category.name} (limit: {max_videos} videos per label)")
        
        # Find real/fake folders
        label_folders = [d for d in category.iterdir() if d.is_dir()]
        
        for label_folder in label_folders:
            label = label_folder.name  # 'real' or 'fake'
            
            # Get all video files
            video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.MOV']
            video_files = []
            for ext in video_extensions:
                video_files.extend(list(label_folder.glob(f'*{ext}')))
            
            # Randomly sample videos if we have more than the limit
            if len(video_files) > max_videos:
                print(f"Found {len(video_files)} {label} videos in {category.name}, limiting to {max_videos}")
                video_files = random.sample(video_files, max_videos)
            else:
                print(f"Processing all {len(video_files)} {label} videos in {category.name}")
            
            # For benchmarking
            start_time = time.time()
            processed_count = 0
            
            # Process each video with progress bar
            for video_file in tqdm(video_files, desc=f"{category.name}/{label}"):
                try:
                    # Use CUDA for processing if available
                    with torch.no_grad():
                        features = extract_facial_flicker_features(
                            str(video_file),
                            face_app,
                            sample_frames,
                            use_cuda=use_cuda
                        )
                    
                    if features:
                        features['category'] = category.name
                        features['label'] = label
                        results.append(features)
                        processed_count += 1
                        
                        # Periodically save results and flush GPU memory
                        if processed_count % 5 == 0 and cuda_available:
                            torch.cuda.empty_cache()
                            
                except Exception as e:
                    print(f"Error processing {video_file}: {e}")
            
            # Show processing stats
            elapsed_time = time.time() - start_time
            if processed_count > 0:
                print(f"Processed {processed_count} videos in {elapsed_time:.2f} seconds")
                print(f"Average processing time: {elapsed_time/processed_count:.2f} seconds per video")
                
            # Clear GPU memory after each folder
            if cuda_available:
                torch.cuda.empty_cache()
    
    # Create DataFrame and save to CSV
    if results:
        df = pd.DataFrame(results)
        df.to_csv(output_file, index=False)
        print(f"Features saved to {output_file}")
        
        # Display some statistics
        print("\nFeature Statistics:")
        for label in ['real', 'fake']:
            label_df = df[df['label'] == label]
            if not label_df.empty:
                print(f"\n{label.upper()} Videos:")
                print(f"  Number of videos: {len(label_df)}")
                print(f"  Training videos: {len(label_df[label_df['category'] == 'train'])}")
                print(f"  Testing videos: {len(label_df[label_df['category'] == 'test'])}")
                for feature in ['face_temporal_variance', 'face_flicker_density', 'face_normalized_variance']:
                    if feature in label_df.columns:
                        print(f"  {feature}: mean={label_df[feature].mean():.4f}, std={label_df[feature].std():.4f}")
        
        return df
    else:
        print("No results to save")
        return None

def visualize_features(df, output_folder):
    """
    Create visualizations of the extracted features.
    
    Args:
        df (DataFrame): DataFrame containing extracted features
        output_folder (str): Folder to save visualizations
    """
    os.makedirs(output_folder, exist_ok=True)
    
    # Key features to visualize
    features_to_plot = [
        'face_temporal_variance', 
        'face_temporal_std', 
        'face_flicker_density', 
        'face_normalized_variance',
        'face_mean_abs_diff',
        'face_spatial_flicker_mean',
        'face_flicker_concentration'
    ]
    
    # Filter to only include features that exist in the dataframe
    features_to_plot = [f for f in features_to_plot if f in df.columns]
    
    # Create histograms comparing real vs fake
    for feature in features_to_plot:
        plt.figure(figsize=(10, 6))
        
        real_data = df[df['label'] == 'real'][feature].dropna()
        fake_data = df[df['label'] == 'fake'][feature].dropna()
        
        if len(real_data) > 0 and len(fake_data) > 0:
            # Calculate optimal number of bins based on data
            bins = min(30, max(10, int(np.sqrt(len(df)))))
            
            plt.hist(real_data, bins=bins, alpha=0.7, label='Real', density=True)
            plt.hist(fake_data, bins=bins, alpha=0.7, label='Fake', density=True)
            
            plt.title(f'Distribution of {feature}')
            plt.xlabel(feature)
            plt.ylabel('Density')
            plt.legend()
            plt.grid(alpha=0.3)
            plt.tight_layout()
            
            plt.savefig(os.path.join(output_folder, f'{feature}_histogram.png'))
            plt.close()
    
    # Create scatter plots to see relationships between features
    if len(features_to_plot) >= 2:
        for i, feature1 in enumerate(features_to_plot[:-1]):
            for feature2 in features_to_plot[i+1:]:
                plt.figure(figsize=(10, 6))
                
                real_df = df[df['label'] == 'real']
                fake_df = df[df['label'] == 'fake']
                
                plt.scatter(real_df[feature1], real_df[feature2], alpha=0.7, label='Real')
                plt.scatter(fake_df[feature1], fake_df[feature2], alpha=0.7, label='Fake')
                
                plt.title(f'{feature1} vs {feature2}')
                plt.xlabel(feature1)
                plt.ylabel(feature2)
                plt.legend()
                plt.grid(alpha=0.3)
                plt.tight_layout()
                
                plt.savefig(os.path.join(output_folder, f'{feature1}_vs_{feature2}.png'))
                plt.close()
    
    # Create t-SNE visualization
    create_tsne_visualization(df, output_folder)
    
    # Create UMAP visualization if available
    if UMAP_AVAILABLE:
        create_umap_visualization(df, output_folder)
    else:
        print("UMAP not installed. Skipping UMAP visualization.")

def create_tsne_visualization(df, output_folder):
    """
    Create t-SNE visualization of the features.
    
    Args:
        df (DataFrame): DataFrame containing extracted features
        output_folder (str): Folder to save visualizations
    """
    print("Creating t-SNE visualization...")
    
    # Select numerical features for t-SNE
    feature_columns = [col for col in df.columns if col not in ['video_path', 'category', 'label']]
    
    # Drop any rows with NaN values
    df_clean = df.dropna(subset=feature_columns)
    
    if len(df_clean) < 3:
        print("Not enough data points for t-SNE visualization")
        return
    
    # Extract features and standardize
    X = df_clean[feature_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Perform t-SNE dimensionality reduction
    # Use lower perplexity for small datasets
    perplexity = min(30, max(5, len(X_scaled) // 5))
    
    try:
        tsne = TSNE(n_components=2, perplexity=perplexity, random_state=42)
        X_tsne = tsne.fit_transform(X_scaled)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        real_indices = df_clean['label'] == 'real'
        fake_indices = df_clean['label'] == 'fake'
        
        plt.scatter(X_tsne[real_indices, 0], X_tsne[real_indices, 1], 
                   alpha=0.8, label='Real', s=50)
        plt.scatter(X_tsne[fake_indices, 0], X_tsne[fake_indices, 1], 
                   alpha=0.8, label='Fake', s=50)
        
        plt.title('t-SNE Visualization of Facial Temporal Flicker Features')
        plt.xlabel('t-SNE Dimension 1')
        plt.ylabel('t-SNE Dimension 2')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_folder, 'tsne_visualization.png'))
        plt.close()
        print("t-SNE visualization saved.")
        
    except Exception as e:
        print(f"Error creating t-SNE visualization: {e}")

def create_umap_visualization(df, output_folder):
    """
    Create UMAP visualization of the features.
    
    Args:
        df (DataFrame): DataFrame containing extracted features
        output_folder (str): Folder to save visualizations
    """
    print("Creating UMAP visualization...")
    
    # Select numerical features for UMAP
    feature_columns = [col for col in df.columns if col not in ['video_path', 'category', 'label']]
    
    # Drop any rows with NaN values
    df_clean = df.dropna(subset=feature_columns)
    
    if len(df_clean) < 3:
        print("Not enough data points for UMAP visualization")
        return
    
    # Extract features and standardize
    X = df_clean[feature_columns].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Adjust n_neighbors based on dataset size
    n_neighbors = min(15, max(2, len(X_scaled) // 10))
    
    try:
        # Perform UMAP dimensionality reduction
        reducer = umap.UMAP(n_neighbors=n_neighbors, min_dist=0.1, 
                          n_components=2, random_state=42)
        X_umap = reducer.fit_transform(X_scaled)
        
        # Create visualization
        plt.figure(figsize=(12, 10))
        
        real_indices = df_clean['label'] == 'real'
        fake_indices = df_clean['label'] == 'fake'
        
        plt.scatter(X_umap[real_indices, 0], X_umap[real_indices, 1], 
                   alpha=0.8, label='Real', s=50)
        plt.scatter(X_umap[fake_indices, 0], X_umap[fake_indices, 1], 
                   alpha=0.8, label='Fake', s=50)
        
        plt.title('UMAP Visualization of Facial Temporal Flicker Features')
        plt.xlabel('UMAP Dimension 1')
        plt.ylabel('UMAP Dimension 2')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        
        # Save the figure
        plt.savefig(os.path.join(output_folder, 'umap_visualization.png'))
        plt.close()
        print("UMAP visualization saved.")
        
    except Exception as e:
        print(f"Error creating UMAP visualization: {e}")

def main():
    """Main function to run the feature extraction."""
    # Create output directories if they don't exist
    os.makedirs("./output_face", exist_ok=True)
    os.makedirs("./visualizations_face", exist_ok=True)
    
    parser = argparse.ArgumentParser(description='Extract facial temporal flicker features from videos for deepfake detection')
    
    parser.add_argument('--root', type=str, default='C:\\Users\\athar\\Documents\\GitHub\\testcv\\finalData',
                        help='Root folder containing test and train folders')
    parser.add_argument('--output', type=str, default='./output_face/facial_flicker_features.csv',
                        help='Output CSV file path')
    parser.add_argument('--vis_output', type=str, default='./visualizations_face',
                        help='Output folder for visualizations')
    parser.add_argument('--sample_frames', type=int, default=None,
                        help='Number of frames to sample per video (None for all frames)')
    parser.add_argument('--use_cuda', action='store_true', default=True,
                        help='Use CUDA acceleration for video processing (only affects PyTorch, not InsightFace)')
    parser.add_argument('--no_cuda', dest='use_cuda', action='store_false',
                        help='Disable CUDA acceleration')
    parser.add_argument('--max_train', type=int, default=100,
                        help='Maximum number of training videos per label (real/fake)')
    parser.add_argument('--max_test', type=int, default=20,
                        help='Maximum number of testing videos per label (real/fake)')
    parser.add_argument('--face_padding', type=float, default=0.2,
                        help='Padding around detected faces (as fraction of face size)')
    parser.add_argument('--single_video', type=str, default=None,
                        help='Path to a single video to analyze (bypasses dataset processing)')
    
    args = parser.parse_args()
    
    print(f"Processing videos from: {args.root}")
    print(f"Output will be saved to: {args.output}")
    print(f"CUDA enabled for PyTorch: {args.use_cuda}")
    print(f"Max training videos per label: {args.max_train}")
    print(f"Max testing videos per label: {args.max_test}")
    
    # Check InsightFace availability
    if not INSIGHTFACE_AVAILABLE:
        print("ERROR: InsightFace is required but not installed.")
        print("Please install with: pip install insightface")
        return
    
    # Check CUDA availability before processing (only for PyTorch, not InsightFace)
    if args.use_cuda:
        if torch.cuda.is_available():
            print(f"CUDA device available for PyTorch: {torch.cuda.get_device_name(0)}")
            # Set optimal thread settings for CUDA
            torch.set_num_threads(4)  # Limit CPU threads when using GPU
        else:
            print("CUDA requested but not available. PyTorch falling back to CPU.")
            args.use_cuda = False
    
    # Initialize face detector (always CPU-only)
    print("Initializing InsightFace face detector (CPU only)...")
    face_app = initialize_face_detector(use_cuda=False)
    
    # If a single video is specified, analyze just that video
    if args.single_video:
        if os.path.exists(args.single_video):
            print(f"Analyzing single video: {args.single_video}")
            features = analyze_single_video(
                args.single_video,
                output_dir=args.vis_output,
                use_cuda=args.use_cuda,
                sample_frames=args.sample_frames,
                face_padding=args.face_padding
            )
            if features:
                # Save features to CSV
                df = pd.DataFrame([features])
                os.makedirs(os.path.dirname(args.output), exist_ok=True)
                df.to_csv(args.output, index=False)
                print(f"Features saved to {args.output}")
        else:
            print(f"ERROR: Specified video file does not exist: {args.single_video}")
    else:
        # Process dataset and get features
        df = process_dataset(
            args.root, 
            args.output, 
            face_app,
            args.sample_frames,
            use_cuda=args.use_cuda,
            max_train_videos=args.max_train,
            max_test_videos=args.max_test
        )
        
        # Create visualizations if data was successfully processed
        if df is not None and not df.empty:
            visualize_features(df, args.vis_output)
            print(f"Visualizations saved to: {args.vis_output}")

# Add a function for face flicker analysis on a single video
def analyze_single_video(video_path, output_dir="./output_face", use_cuda=True, sample_frames=None, face_padding=0.2):
    """
    Analyze a single video for facial temporal flicker and visualize the results.
    
    Args:
        video_path (str): Path to the video file
        output_dir (str): Directory to save output visualizations
        use_cuda (bool): Whether to use CUDA acceleration for PyTorch processing
        sample_frames (int): Number of frames to sample
        face_padding (float): Padding around face region
        
    Returns:
        dict: Dictionary of extracted features
    """
    if not os.path.exists(video_path):
        print(f"Error: Video file not found: {video_path}")
        return None
    
    if output_dir is not None:
        os.makedirs(output_dir, exist_ok=True)
    
    # Initialize face detector (CPU only)
    print("Initializing InsightFace face detector (CPU only)...")
    face_app = initialize_face_detector(use_cuda=False)
    
    # Extract features
    print(f"Analyzing video: {video_path}")
    features = extract_facial_flicker_features(
        video_path,
        face_app,
        sample_frames=sample_frames,
        use_cuda=use_cuda
    )
    
    if features is None:
        print("Failed to extract features. No faces detected or video processing failed.")
        return None
    
    # Print features
    print("\nExtracted Facial Flicker Features:")
    for feature, value in features.items():
        if feature != 'video_path':
            print(f"  {feature}: {value}")
    
    # Create visualizations if output directory is provided
    if output_dir is not None:
        # Create overlaid video showing flicker regions
        create_flicker_visualization_video(
            video_path,
            face_app,
            os.path.join(output_dir, "flicker_visualization.mp4"),
            use_cuda=use_cuda,
            face_padding=face_padding
        )
    
    return features

def create_flicker_visualization_video(video_path, face_app, output_path="./visualizations_face/flicker_visualization.mp4", use_cuda=True, face_padding=0.2):
    """
    Create a visualization video showing the facial flicker regions.
    
    Args:
        video_path (str): Path to the input video
        face_app (FaceAnalysis): InsightFace face detector
        output_path (str): Path to save the output video
        use_cuda (bool): Whether to use CUDA for processing
        face_padding (float): Padding around face region
    """
    print(f"Creating flicker visualization video: {output_path}")
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file: {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # Create video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    # Initialize variables for flicker detection
    prev_face_gray = None
    face_bbox = None
    max_diff_val = 0
    
    # Process frames
    with tqdm(total=total_frames, desc="Creating visualization") as pbar:
        frame_idx = 0
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            
            # Detect face in the first frame
            if frame_idx == 0 or face_bbox is None:
                face_regions = extract_face_regions(frame, face_app, padding=face_padding)
                if face_regions:
                    # Use the largest face if multiple are detected
                    largest_face_idx = 0
                    largest_area = 0
                    for i, face_data in enumerate(face_regions):
                        x1, y1, x2, y2 = face_data['bbox']
                        area = (x2 - x1) * (y2 - y1)
                        if area > largest_area:
                            largest_area = area
                            largest_face_idx = i
                    
                    face_bbox = face_regions[largest_face_idx]['bbox']
            
            # Process frame if face was detected
            if face_bbox is not None:
                x1, y1, x2, y2 = face_bbox
                face_region = frame[y1:y2, x1:x2]
                
                if face_region.size > 0:
                    # Convert to grayscale
                    face_gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
                    
                    # Calculate difference map if we have a previous frame
                    if prev_face_gray is not None and prev_face_gray.shape == face_gray.shape:
                        # Calculate absolute difference
                        diff_map = cv2.absdiff(face_gray, prev_face_gray)
                        
                        # Update max difference value for normalization
                        current_max = np.max(diff_map)
                        if current_max > max_diff_val:
                            max_diff_val = current_max
                        
                        # Normalize and colorize difference map for visualization
                        if max_diff_val > 0:
                            # Normalize to 0-255 range
                            norm_diff = (diff_map / max_diff_val * 255).astype(np.uint8)
                            
                            # Apply color map for better visualization
                            color_diff = cv2.applyColorMap(norm_diff, cv2.COLORMAP_JET)
                            
                            # Draw face bounding box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            
                            # Overlay difference map on original frame
                            overlay = frame.copy()
                            overlay[y1:y2, x1:x2] = cv2.addWeighted(
                                frame[y1:y2, x1:x2], 0.3, 
                                color_diff, 0.7, 0
                            )
                            
                            # Add text with max difference value
                            cv2.putText(
                                overlay, f"Max Diff: {current_max:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2
                            )
                            
                            # Write the frame
                            out.write(overlay)
                        else:
                            # If no difference, just write the original frame with face box
                            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                            out.write(frame)
                    else:
                        # For the first frame, just draw the face box
                        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                        out.write(frame)
                    
                    # Update previous frame
                    prev_face_gray = face_gray
                else:
                    # If face region is empty, just write the original frame
                    out.write(frame)
            else:
                # If no face detected, just write the original frame
                out.write(frame)
            
            frame_idx += 1
            pbar.update(1)
    
    # Release resources
    cap.release()
    out.release()
    print(f"Visualization saved to: {output_path}")

if __name__ == '__main__':
    main()