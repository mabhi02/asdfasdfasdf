import os
import numpy as np
import cv2
import torch
import insightface
from insightface.app import FaceAnalysis
from sklearn.manifold import TSNE
from sklearn.ensemble import RandomForestClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import pandas as pd
from scipy import signal
from scipy.stats import pearsonr
import random

# Check for CUDA availability for torch operations
cuda_available = torch.cuda.is_available()
print(f"CUDA Available: {cuda_available}")
if cuda_available:
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    # Create a CUDA tensor for verification
    test_tensor = torch.zeros(1).cuda()
    print(f"Test tensor device: {test_tensor.device}")

# Set random seed for reproducibility
random_seed = 42
np.random.seed(random_seed)
random.seed(random_seed)
torch.manual_seed(random_seed)
if cuda_available:
    torch.cuda.manual_seed_all(random_seed)

# Define paths
base_path = r"C:\Users\athar\Documents\GitHub\testcv\finalData"
train_fake_path = os.path.join(base_path, "train", "fake")
train_real_path = os.path.join(base_path, "train", "real")
test_fake_path = os.path.join(base_path, "test", "fake")
test_real_path = os.path.join(base_path, "test", "real")

# Initialize InsightFace (CPU mode for compatibility, but could be GPU)
face_analyzer = FaceAnalysis(name='buffalo_l', providers=['CPUExecutionProvider'])
face_analyzer.prepare(ctx_id=0, det_size=(640, 640))

def get_video_files(directory, limit=None, seed=None, exclude_files=None):
    """Get all video files from directory with consistent selection using seed
    
    Args:
        directory: Directory containing video files
        limit: Maximum number of files to return
        seed: Random seed for consistent selection
        exclude_files: List of filenames to exclude (to avoid duplicates)
    """
    video_files = []
    for file in os.listdir(directory):
        if file.endswith(('.mp4', '.avi', '.mov', '.MOV')):
            full_path = os.path.join(directory, file)
            # Skip files that should be excluded
            if exclude_files and os.path.basename(full_path) in exclude_files:
                continue
            video_files.append(full_path)
    
    # Sort to ensure deterministic order before selection
    video_files.sort()
    
    # Use the provided seed for consistent selection
    if seed is not None:
        # Create a separate random generator instance with this seed
        # to avoid affecting the global random state
        rng = random.Random(seed)
        rng.shuffle(video_files)
    
    if limit and len(video_files) > limit:
        return video_files[:limit]
    return video_files

def extract_face_landmarks(video_path, target_fps=2, max_frames=60):
    """Extract face landmarks from video frames at a rate of 2 frames per second with an upper limit
    
    Args:
        video_path: Path to the video file
        target_fps: Target frames per second for sampling (default: 2)
        max_frames: Maximum number of frames to process (default: 60, equivalent to 30 seconds at 2 FPS)
    """
    cap = cv2.VideoCapture(video_path)
    
    # Get video properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = frame_count / fps if fps > 0 else 0
    
    # Calculate sampling interval (frames to skip for target_fps sampling)
    sampling_interval = max(1, int(fps / target_fps))
    
    print(f"Video: {os.path.basename(video_path)} | Duration: {duration:.2f}s | "
          f"Original FPS: {fps:.2f} | Sampling interval: {sampling_interval} frames "
          f"(targeting {target_fps} fps, max {max_frames} frames)")
    
    frames = []
    landmarks_sequence = []
    current_frame = 0
    processed_frames = 0
    
    while True:
        ret, frame = cap.read()
        if not ret or processed_frames >= max_frames:
            break
            
        # Only process frames at the desired sampling rate
        if current_frame % sampling_interval == 0:
            # Convert to RGB for InsightFace
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
            
            # Detect faces - InsightFace remains on CPU as specified
            faces = face_analyzer.get(frame_rgb)
            
            if len(faces) > 0:
                # Use the face with the highest detection score
                main_face = max(faces, key=lambda x: x.det_score)
                
                # Get facial landmarks
                landmarks = main_face.landmark_2d_106
                landmarks_sequence.append(landmarks)
            else:
                # If no face detected, pad with zeros
                landmarks_sequence.append(np.zeros((106, 2)))
                
            processed_frames += 1
        
        current_frame += 1
    
    cap.release()
    
    # Log frame limits if applied
    if processed_frames >= max_frames:
        print(f"  Note: Limited to {max_frames} frames (original video had {frame_count} frames)")
    
    # Convert to numpy array
    landmarks_sequence = np.array(landmarks_sequence)
    return frames, landmarks_sequence

def define_facial_regions():
    """Define indices for different facial regions based on InsightFace's 106-point landmarks"""
    # These indices are based on InsightFace's 106-point landmark system
    regions = {
        'lips': list(range(46, 64)) + list(range(76, 96)),  # Upper and lower lips
        'left_cheek': [0, 1, 2, 3, 4, 31, 32, 33, 41, 42, 50, 51, 52],
        'right_cheek': [12, 13, 14, 15, 16, 35, 36, 37, 45, 46, 54, 55, 56],
        'jaw': list(range(4, 13))  # Jawline points
    }
    return regions

def compute_region_motion(landmarks_sequence, region_indices):
    """Compute motion for a specific facial region with CUDA acceleration for critical computations"""
    region_landmarks = landmarks_sequence[:, region_indices, :]
    
    if cuda_available:
        # Transfer to GPU using torch
        region_landmarks_gpu = torch.tensor(region_landmarks, device='cuda')
        
        # Calculate frame-to-frame motion (displacement)
        motion_gpu = torch.zeros((len(landmarks_sequence)-1, len(region_indices), 2), device='cuda')
        
        for i in range(len(landmarks_sequence)-1):
            motion_gpu[i] = region_landmarks_gpu[i+1] - region_landmarks_gpu[i]
        
        # Compute the average motion magnitude for the region
        motion_squared = torch.sum(motion_gpu**2, dim=2)
        motion_magnitude_gpu = torch.sqrt(motion_squared).mean(dim=1)
        
        # Transfer back to CPU
        motion = motion_gpu.cpu().numpy()
        motion_magnitude = motion_magnitude_gpu.cpu().numpy()
    else:
        # CPU implementation
        motion = np.zeros((len(landmarks_sequence)-1, len(region_indices), 2))
        
        for i in range(len(landmarks_sequence)-1):
            motion[i] = region_landmarks[i+1] - region_landmarks[i]
        
        # Compute the average motion magnitude for the region
        motion_magnitude = np.sqrt(np.sum(motion**2, axis=2)).mean(axis=1)
    
    return motion, motion_magnitude

def extract_motion_features(landmarks_sequence):
    """Extract all 18 motion synchronization features from landmarks sequence using CUDA for core computations"""
    regions = define_facial_regions()
    
    # Skip videos with too few frames - adjusted for 2fps sampling
    # At 2fps, 10 seconds of video would have 20 frames
    if len(landmarks_sequence) < 10:
        print(f"Skipping video with only {len(landmarks_sequence)} frames (too few for analysis)")
        return None
    
    # 1. Extract motion for each region (CUDA-accelerated)
    lips_motion, lips_magnitude = compute_region_motion(landmarks_sequence, regions['lips'])
    left_cheek_motion, left_cheek_magnitude = compute_region_motion(landmarks_sequence, regions['left_cheek'])
    right_cheek_motion, right_cheek_magnitude = compute_region_motion(landmarks_sequence, regions['right_cheek'])
    jaw_motion, jaw_magnitude = compute_region_motion(landmarks_sequence, regions['jaw'])
    
    features = []
    
    # 2. Motion Correlation Features (3)
    lip_left_corr = pearsonr(lips_magnitude, left_cheek_magnitude)[0]
    lip_right_corr = pearsonr(lips_magnitude, right_cheek_magnitude)[0]
    lip_jaw_corr = pearsonr(lips_magnitude, jaw_magnitude)[0]
    
    # Handle NaN values
    lip_left_corr = 0 if np.isnan(lip_left_corr) else lip_left_corr
    lip_right_corr = 0 if np.isnan(lip_right_corr) else lip_right_corr
    lip_jaw_corr = 0 if np.isnan(lip_jaw_corr) else lip_jaw_corr
    
    features.extend([lip_left_corr, lip_right_corr, lip_jaw_corr])
    
    # 3. Movement Magnitude Ratio Features (3) - CUDA-accelerated
    if cuda_available:
        # Transfer to GPU
        lips_magnitude_gpu = torch.tensor(lips_magnitude, device='cuda')
        left_cheek_magnitude_gpu = torch.tensor(left_cheek_magnitude, device='cuda')
        right_cheek_magnitude_gpu = torch.tensor(right_cheek_magnitude, device='cuda')
        jaw_magnitude_gpu = torch.tensor(jaw_magnitude, device='cuda')
        
        lip_left_ratio = torch.mean(lips_magnitude_gpu) / (torch.mean(left_cheek_magnitude_gpu) + 1e-6)
        lip_right_ratio = torch.mean(lips_magnitude_gpu) / (torch.mean(right_cheek_magnitude_gpu) + 1e-6)
        lip_jaw_ratio = torch.mean(lips_magnitude_gpu) / (torch.mean(jaw_magnitude_gpu) + 1e-6)
        
        # Transfer back to CPU
        lip_left_ratio = lip_left_ratio.cpu().item()
        lip_right_ratio = lip_right_ratio.cpu().item()
        lip_jaw_ratio = lip_jaw_ratio.cpu().item()
    else:
        lip_left_ratio = np.mean(lips_magnitude) / (np.mean(left_cheek_magnitude) + 1e-6)
        lip_right_ratio = np.mean(lips_magnitude) / (np.mean(right_cheek_magnitude) + 1e-6)
        lip_jaw_ratio = np.mean(lips_magnitude) / (np.mean(jaw_magnitude) + 1e-6)
    
    features.extend([lip_left_ratio, lip_right_ratio, lip_jaw_ratio])
    
    # 4. Temporal Lag Features (3) - Using cross-correlation to find lag
    def find_lag(signal1, signal2, max_lag=10):
        if cuda_available:
            # Transfer to GPU
            signal1_gpu = torch.tensor(signal1, device='cuda')
            signal2_gpu = torch.tensor(signal2, device='cuda')
            
            # For cross-correlation, we'll use the CPU version and just accelerate the signal processing
            # since PyTorch's correlation functions are more complex to use
            signal1_processed = signal1_gpu.cpu().numpy()
            signal2_processed = signal2_gpu.cpu().numpy()
            cross_corr = np.correlate(signal1_processed, signal2_processed, mode='full')
            max_idx = np.argmax(cross_corr)
            lag = max_idx - (len(signal1) - 1)
        else:
            cross_corr = np.correlate(signal1, signal2, mode='full')
            max_idx = np.argmax(cross_corr)
            lag = max_idx - (len(signal1) - 1)
            
        return min(max(lag, -max_lag), max_lag)  # Limit to ±max_lag frames
    
    lip_left_lag = find_lag(lips_magnitude, left_cheek_magnitude)
    lip_right_lag = find_lag(lips_magnitude, right_cheek_magnitude)
    lip_jaw_lag = find_lag(lips_magnitude, jaw_magnitude)
    
    features.extend([lip_left_lag, lip_right_lag, lip_jaw_lag])
    
    # 5. Motion Consistency Features (3)
    # Calculate correlations over sliding windows
    # Window size adjusted to 5 frames at 2fps (represents ~2.5 seconds of video)
    window_size = 5
    window_corrs_left = []
    window_corrs_jaw = []
    consistent_frames = 0
    
    for i in range(len(lips_magnitude) - window_size):
        lip_window = lips_magnitude[i:i+window_size]
        left_window = left_cheek_magnitude[i:i+window_size]
        jaw_window = jaw_magnitude[i:i+window_size]
        
        window_corr_left = pearsonr(lip_window, left_window)[0]
        window_corr_jaw = pearsonr(lip_window, jaw_window)[0]
        
        if not np.isnan(window_corr_left):
            window_corrs_left.append(window_corr_left)
        if not np.isnan(window_corr_jaw):
            window_corrs_jaw.append(window_corr_jaw)
            
        # Count frames with consistent motion (correlation > 0.5)
        if window_corr_left > 0.5 and window_corr_jaw > 0.5:
            consistent_frames += 1
            
    std_lip_cheek_corr = np.std(window_corrs_left) if window_corrs_left else 0
    std_lip_jaw_corr = np.std(window_corrs_jaw) if window_corrs_jaw else 0
    consistency_ratio = consistent_frames / (len(lips_magnitude) - window_size) if len(lips_magnitude) > window_size else 0
    
    features.extend([std_lip_cheek_corr, std_lip_jaw_corr, consistency_ratio])
    
    # 6. Energy Distribution Features (3) - CUDA-accelerated
    if cuda_available:
        lips_magnitude_gpu = torch.tensor(lips_magnitude, device='cuda')
        left_cheek_magnitude_gpu = torch.tensor(left_cheek_magnitude, device='cuda')
        right_cheek_magnitude_gpu = torch.tensor(right_cheek_magnitude, device='cuda')
        jaw_magnitude_gpu = torch.tensor(jaw_magnitude, device='cuda')
        
        total_energy = torch.sum(lips_magnitude_gpu) + torch.sum(left_cheek_magnitude_gpu) + \
                       torch.sum(right_cheek_magnitude_gpu) + torch.sum(jaw_magnitude_gpu)
        lip_energy_ratio = torch.sum(lips_magnitude_gpu) / (total_energy + 1e-6)
        cheek_energy_ratio = (torch.sum(left_cheek_magnitude_gpu) + torch.sum(right_cheek_magnitude_gpu)) / (total_energy + 1e-6)
        jaw_energy_ratio = torch.sum(jaw_magnitude_gpu) / (total_energy + 1e-6)
        
        # Transfer back to CPU
        lip_energy_ratio = lip_energy_ratio.cpu().item()
        cheek_energy_ratio = cheek_energy_ratio.cpu().item()
        jaw_energy_ratio = jaw_energy_ratio.cpu().item()
    else:
        total_energy = np.sum(lips_magnitude) + np.sum(left_cheek_magnitude) + np.sum(right_cheek_magnitude) + np.sum(jaw_magnitude)
        lip_energy_ratio = np.sum(lips_magnitude) / (total_energy + 1e-6)
        cheek_energy_ratio = (np.sum(left_cheek_magnitude) + np.sum(right_cheek_magnitude)) / (total_energy + 1e-6)
        jaw_energy_ratio = np.sum(jaw_magnitude) / (total_energy + 1e-6)
    
    features.extend([lip_energy_ratio, cheek_energy_ratio, jaw_energy_ratio])
    
    # 7. Frequency-Based Features (3)
    def get_dominant_freq(signal_data):
        if len(signal_data) < 4:  # Need minimum data for FFT
            return 0
        
        if cuda_available:
            # Transfer to GPU
            signal_data_gpu = torch.tensor(signal_data, device='cuda')
            
            # Remove mean - GPU accelerated
            signal_data_gpu = signal_data_gpu - torch.mean(signal_data_gpu)
            
            # FFT needs to be done on CPU as torch.fft is more complex for this case
            signal_processed = signal_data_gpu.cpu().numpy()
            
            # Remove mean and perform FFT
            fft = np.abs(np.fft.rfft(signal_processed))
            freqs = np.fft.rfftfreq(len(signal_processed))
            
            # Find dominant frequency (exclude DC component)
            if len(fft) > 1:
                dom_freq_idx = np.argmax(fft[1:]) + 1
                return freqs[dom_freq_idx]
        else:
            # Remove mean and perform FFT
            signal_data = signal_data - np.mean(signal_data)
            fft = np.abs(np.fft.rfft(signal_data))
            freqs = np.fft.rfftfreq(len(signal_data))
            
            # Find dominant frequency (exclude DC component)
            if len(fft) > 1:
                dom_freq_idx = np.argmax(fft[1:]) + 1
                return freqs[dom_freq_idx]
        return 0
    
    # Calculate dominant frequencies
    lip_dom_freq = get_dominant_freq(lips_magnitude)
    cheek_dom_freq = get_dominant_freq(left_cheek_magnitude)
    jaw_dom_freq = get_dominant_freq(jaw_magnitude)
    
    # Calculate frequency ratios and phase difference
    lip_cheek_freq_ratio = lip_dom_freq / (cheek_dom_freq + 1e-6)
    lip_jaw_freq_ratio = lip_dom_freq / (jaw_dom_freq + 1e-6)
    
    # Calculate phase difference
    if len(lips_magnitude) > 0 and len(jaw_magnitude) > 0:
        # Cross-correlation for phase difference
        corr = signal.correlate(lips_magnitude, jaw_magnitude, mode='same')
        max_idx = np.argmax(corr)
        phase_diff = 2 * np.pi * (max_idx - len(corr)//2) / len(corr)
    else:
        phase_diff = 0
        
    features.extend([lip_cheek_freq_ratio, lip_jaw_freq_ratio, phase_diff])
    
    # Clear CUDA cache to prevent memory issues
    if cuda_available:
        torch.cuda.empty_cache()
    
    return np.array(features)

def process_videos(video_paths, label, target_fps=2, max_frames=60):
    """Process a list of videos and extract features"""
    features = []
    labels = []
    filenames = []
    landmarks_data = {}  # Dictionary to store landmarks for each video
    
    for video_path in tqdm(video_paths, desc=f"Processing {label} videos"):
        try:
            # Extract face landmarks using InsightFace with frame limits
            _, landmarks_sequence = extract_face_landmarks(video_path, target_fps=target_fps, max_frames=max_frames)
            
            # Store landmarks for this video
            video_filename = os.path.basename(video_path)
            landmarks_data[video_filename] = landmarks_sequence
            
            # CUDA-accelerated feature extraction for the core calculations
            video_features = extract_motion_features(landmarks_sequence)
            
            if video_features is not None:
                features.append(video_features)
                labels.append(1 if label == 'fake' else 0)  # 1 for fake, 0 for real
                filenames.append(video_filename)
        except Exception as e:
            print(f"Error processing {video_path}: {e}")
    
    return np.array(features) if features else np.array([]), np.array(labels), filenames, landmarks_data

# Modify the visualize_tsne function
def visualize_tsne(features, labels, filenames, save_path):
    """Apply t-SNE and visualize the results"""
    # Create the visualization directory if it doesn't exist
    vis_dir = "v2_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Update the save path to be in the visualizations directory
    save_path = os.path.join(vis_dir, os.path.basename(save_path))
    
    # Handle case with small number of samples
    n_samples = features.shape[0]
    
    # Adjust perplexity based on sample size - perplexity must be less than n_samples
    if n_samples <= 5:
        print(f"Warning: Not enough samples ({n_samples}) for t-SNE visualization in {save_path}. Skipping.")
        return
    
    # Set perplexity to min(30, n_samples/3) with a minimum of 2
    perplexity = min(30, max(2, n_samples // 3))
    print(f"Using perplexity of {perplexity} for {n_samples} samples in {save_path}")
    
    # Apply t-SNE using scikit-learn (CPU version)
    tsne = TSNE(n_components=2, perplexity=perplexity, learning_rate=200, n_iter=1000, random_state=random_seed)
    features_embedded = tsne.fit_transform(features)
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame({
        'x': features_embedded[:, 0],
        'y': features_embedded[:, 1],
        'label': ['AI-Generated' if l == 1 else 'Real' for l in labels],
        'filename': filenames
    })
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='x', y='y', hue='label', style='label', s=100)
    
    # Add labels for potentially interesting points
    for i, row in df.iterrows():
        # Add labels for some points (e.g., outliers)
        if (abs(row['x']) > np.percentile(abs(df['x']), 90)) or (abs(row['y']) > np.percentile(abs(df['y']), 90)):
            plt.text(row['x'], row['y'], row['filename'], fontsize=8)
    
    plt.title('t-SNE Visualization of Facial Motion Synchronization Features')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

# Modify the visualize_pca function
def visualize_pca(features, labels, filenames, save_path):
    """Apply PCA and visualize the results"""
    # Create the visualization directory if it doesn't exist
    vis_dir = "v2_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Update the save path to be in the visualizations directory
    save_path = os.path.join(vis_dir, os.path.basename(save_path))
    
    # Handle case with small number of samples
    n_samples = features.shape[0]
    n_features = features.shape[1]
    
    # PCA needs at least 2 samples
    if n_samples < 2:
        print(f"Warning: Not enough samples ({n_samples}) for PCA visualization in {save_path}. Skipping.")
        return None
    
    # Choose appropriate number of components - can't have more components than samples or features
    n_components = min(2, n_samples - 1, n_features)
    print(f"Using {n_components} components for PCA with {n_samples} samples in {save_path}")
    
    # Apply PCA to reduce to n_components dimensions for visualization
    pca = PCA(n_components=n_components, random_state=random_seed)
    features_pca = pca.fit_transform(features)
    
    # Calculate explained variance
    explained_variance = pca.explained_variance_ratio_
    explained_variance_sum = sum(explained_variance)
    
    # Create DataFrame for easier plotting
    df = pd.DataFrame()
    
    # If we have 2D output, create normal scatter plot
    if n_components == 2:
        df['x'] = features_pca[:, 0]
        df['y'] = features_pca[:, 1]
    # If we only have 1D output, create artificial y-axis with small random noise
    elif n_components == 1:
        df['x'] = features_pca[:, 0]
        # Add small random noise for y-axis to visualize
        np.random.seed(random_seed)
        df['y'] = np.random.normal(0, 0.01, size=n_samples)
        plt.ylabel('Random Jitter (for visualization only)')
    
    df['label'] = ['AI-Generated' if l == 1 else 'Real' for l in labels]
    df['filename'] = filenames
    
    # Plot
    plt.figure(figsize=(12, 10))
    sns.scatterplot(data=df, x='x', y='y', hue='label', style='label', s=100)
    
    # Add labels for potentially interesting points
    for i, row in df.iterrows():
        # Add labels for all points if few samples, otherwise just outliers
        if n_samples <= 10 or (abs(row['x']) > np.percentile(abs(df['x']), 90)) or (abs(row['y']) > np.percentile(abs(df['y']), 90)):
            plt.text(row['x'], row['y'], row['filename'], fontsize=8)
    
    # Add explained variance information
    plt.title(f'PCA Visualization of Facial Motion Features\nExplained Variance: {explained_variance_sum:.2f}')
    
    if n_components == 2:
        plt.xlabel(f'PC1 ({explained_variance[0]:.2f} variance explained)')
        plt.ylabel(f'PC2 ({explained_variance[1]:.2f} variance explained)')
    else:
        plt.xlabel(f'PC1 ({explained_variance[0]:.2f} variance explained)')
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()
    
    # If too few samples, skip the detailed analysis
    if n_samples < 5:
        return pca
    
    # Save feature loadings information
    feature_names = [
        'Lip-Left Cheek Corr', 'Lip-Right Cheek Corr', 'Lip-Jaw Corr',
        'Lip-Left Cheek Ratio', 'Lip-Right Cheek Ratio', 'Lip-Jaw Ratio',
        'Lip-Left Lag', 'Lip-Right Lag', 'Lip-Jaw Lag',
        'Lip-Cheek Corr StdDev', 'Lip-Jaw Corr StdDev', 'Motion Consistency',
        'Lip Energy Ratio', 'Cheek Energy Ratio', 'Jaw Energy Ratio',
        'Lip-Cheek Freq Ratio', 'Lip-Jaw Freq Ratio', 'Phase Difference'
    ]


def train_and_evaluate_model(train_features, train_labels, test_features, test_labels):
    """Train a classifier and evaluate performance"""
    # Create the visualization directory if it doesn't exist
    vis_dir = "v2_visualizations"
    os.makedirs(vis_dir, exist_ok=True)
    
    # Create and train the model using scikit-learn's RandomForest
    print("Training RandomForest classifier...")
    clf = RandomForestClassifier(n_estimators=100, random_state=random_seed)
    clf.fit(train_features, train_labels)
    predictions = clf.predict(test_features)
    
    # Print classification report
    print("\nClassification Report:")
    print(classification_report(test_labels, predictions, target_names=['Real', 'AI-Generated']))
    
    # Plot confusion matrix and save to visualization directory
    cm = confusion_matrix(test_labels, predictions)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=['Real', 'AI-Generated'], yticklabels=['Real', 'AI-Generated'])
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'confusion_matrix.png'))
    plt.close()
    
    # Feature importance - save to visualization directory
    feature_names = [
        'Lip-Left Cheek Corr', 'Lip-Right Cheek Corr', 'Lip-Jaw Corr',
        'Lip-Left Cheek Ratio', 'Lip-Right Cheek Ratio', 'Lip-Jaw Ratio',
        'Lip-Left Lag', 'Lip-Right Lag', 'Lip-Jaw Lag',
        'Lip-Cheek Corr StdDev', 'Lip-Jaw Corr StdDev', 'Motion Consistency',
        'Lip Energy Ratio', 'Cheek Energy Ratio', 'Jaw Energy Ratio',
        'Lip-Cheek Freq Ratio', 'Lip-Jaw Freq Ratio', 'Phase Difference'
    ]
    
    importances = clf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(12, 8))
    plt.bar(range(len(importances)), importances[indices])
    plt.xticks(range(len(importances)), [feature_names[i] for i in indices], rotation=90)
    plt.xlabel('Features')
    plt.ylabel('Importance')
    plt.title('Feature Importance')
    plt.tight_layout()
    plt.savefig(os.path.join(vis_dir, 'feature_importance.png'))
    plt.close()
    
    return clf

def analyze_prediction_errors(test_features, test_labels, predictions, filenames):
    """Analyze misclassified videos"""
    errors = test_labels != predictions
    error_indices = np.where(errors)[0]
    
    if len(error_indices) > 0:
        print("\nMisclassified Videos:")
        for idx in error_indices:
            true_label = "Real" if test_labels[idx] == 0 else "AI-Generated"
            pred_label = "Real" if predictions[idx] == 0 else "AI-Generated"
            print(f"File: {filenames[idx]}, True: {true_label}, Predicted: {pred_label}")

def load_features(features_dir="extracted_features", landmarks_dir="extracted_landmarks"):
    """Load previously extracted features and landmarks from disk"""
    if not os.path.exists(features_dir):
        print(f"Error: Feature directory '{features_dir}' not found.")
        return None, None, None, None, None, None, None
    
    print(f"Loading features from '{features_dir}'...")
    
    # Load features from CSV files
    train_df = pd.read_csv(os.path.join(features_dir, "train_features.csv"))
    test_df = pd.read_csv(os.path.join(features_dir, "test_features.csv"))
    
    # Extract components
    train_filenames = train_df['filename'].tolist()
    train_labels = train_df['label'].values
    feature_columns = [col for col in train_df.columns if col not in ['filename', 'label', 'label_text']]
    train_features = train_df[feature_columns].values
    
    test_filenames = test_df['filename'].tolist()
    test_labels = test_df['label'].values
    test_features = test_df[feature_columns].values
    
    # Load landmarks if available
    landmarks = {}
    if os.path.exists(landmarks_dir):
        print(f"Loading landmarks from '{landmarks_dir}'...")
        for file in os.listdir(landmarks_dir):
            if file.endswith('_landmarks.npy'):
                # Extract original filename from landmarks filename
                video_filename = file.replace('_landmarks.npy', '').replace('_', '.', 1)
                landmarks_path = os.path.join(landmarks_dir, file)
                landmarks[video_filename] = np.load(landmarks_path)
        print(f"Loaded landmarks for {len(landmarks)} videos")
    else:
        print(f"Warning: Landmarks directory '{landmarks_dir}' not found.")
    
    print(f"Loaded {len(train_features)} training samples and {len(test_features)} test samples")
    return train_features, train_labels, train_filenames, test_features, test_labels, test_filenames, landmarks

def save_features_to_csv(features, labels, filenames, file_path, feature_names):
    """Save features to a CSV file with descriptive column names"""
    df = pd.DataFrame(features, columns=feature_names)
    df['filename'] = filenames
    df['label'] = labels  # 1 for fake, 0 for real
    df['label_text'] = ['AI-Generated' if l == 1 else 'Real' for l in labels]
    df = df[['filename', 'label', 'label_text'] + feature_names]  # Reorder columns
    df.to_csv(file_path, index=False)
    return df

def save_landmarks(landmarks_data, landmarks_dir):
    """Save facial landmarks to disk"""
    os.makedirs(landmarks_dir, exist_ok=True)
    for video_filename, landmarks in landmarks_data.items():
        # Create a clean filename for the landmarks file
        clean_filename = video_filename.replace('.', '_').replace(' ', '_')
        np.save(os.path.join(landmarks_dir, f"{clean_filename}_landmarks.npy"), landmarks)

def main():
    print("=" * 80)
    print("DEEPFAKE DETECTION USING FACIAL MOTION ANALYSIS - EXPANDED DATASET")
    print("=" * 80)
    print("1. Loading existing features")
    print("2. Processing additional 25 training and 5 test videos")
    print("3. Combining features and training an improved model")
    print("=" * 80)
    
    # Directories for saving features and landmarks
    features_dir = "extracted_features"
    landmarks_dir = "extracted_landmarks"
    
    # Create new directories for the expanded dataset
    expanded_features_dir = "expanded_features"
    expanded_landmarks_dir = "expanded_landmarks"
    os.makedirs(expanded_features_dir, exist_ok=True)
    os.makedirs(expanded_landmarks_dir, exist_ok=True)
    
    # Parameters for frame extraction
    target_fps = 2
    max_frames = 60  # Maximum number of frames per video (30 seconds at 2 FPS)
    
    # Video sample limits
    additional_train_limit = 25  # Additional training videos (both fake and real)
    additional_test_limit = 5    # Additional test videos (both fake and real)
    
    # First, load the existing features
    existing_train_features, existing_train_labels, existing_train_filenames, \
    existing_test_features, existing_test_labels, existing_test_filenames, \
    existing_landmarks = load_features(features_dir, landmarks_dir)
    
    # Check if loading was successful
    if existing_train_features is None:
        print("Failed to load existing features. Make sure the extracted_features directory exists.")
        return
    
    # Copy existing landmarks to expanded landmarks directory
    if existing_landmarks:
        print(f"Copying {len(existing_landmarks)} existing landmarks to expanded directory...")
        save_landmarks(existing_landmarks, expanded_landmarks_dir)
        
    # Feature names for CSV headers
    feature_names = [
        'Lip-Left_Cheek_Corr', 'Lip-Right_Cheek_Corr', 'Lip-Jaw_Corr',
        'Lip-Left_Cheek_Ratio', 'Lip-Right_Cheek_Ratio', 'Lip-Jaw_Ratio',
        'Lip-Left_Lag', 'Lip-Right_Lag', 'Lip-Jaw_Lag',
        'Lip-Cheek_Corr_StdDev', 'Lip-Jaw_Corr_StdDev', 'Motion_Consistency',
        'Lip_Energy_Ratio', 'Cheek_Energy_Ratio', 'Jaw_Energy_Ratio',
        'Lip-Cheek_Freq_Ratio', 'Lip-Jaw_Freq_Ratio', 'Phase_Difference'
    ]
    
    # Process additional videos - choose different ones by using a different seed
    # Use a different seed to get different videos than the original set
    additional_seed = random_seed + 100  # Use a seed offset to get different videos
    
    # Get list of existing filenames to exclude
    existing_fake_train_files = [f for f, l in zip(existing_train_filenames, existing_train_labels) if l == 1]
    existing_real_train_files = [f for f, l in zip(existing_train_filenames, existing_train_labels) if l == 0]
    existing_fake_test_files = [f for f, l in zip(existing_test_filenames, existing_test_labels) if l == 1]
    existing_real_test_files = [f for f, l in zip(existing_test_filenames, existing_test_labels) if l == 0]
    
    print(f"\nGetting additional videos (excluding {len(existing_train_filenames)} existing training and {len(existing_test_filenames)} existing test videos)...")
    
    # Get additional training videos (excluding existing ones)
    additional_train_fake_videos = get_video_files(
        train_fake_path, 
        limit=additional_train_limit, 
        seed=additional_seed, 
        exclude_files=existing_fake_train_files
    )
    
    additional_train_real_videos = get_video_files(
        train_real_path, 
        limit=additional_train_limit, 
        seed=additional_seed, 
        exclude_files=existing_real_train_files
    )
    
    # Get additional test videos (excluding existing ones)
    additional_test_fake_videos = get_video_files(
        test_fake_path, 
        limit=additional_test_limit, 
        seed=additional_seed, 
        exclude_files=existing_fake_test_files
    )
    
    additional_test_real_videos = get_video_files(
        test_real_path, 
        limit=additional_test_limit, 
        seed=additional_seed, 
        exclude_files=existing_real_test_files
    )
    
    print(f"Additional training videos: {len(additional_train_fake_videos)} fake, {len(additional_train_real_videos)} real")
    print(f"Additional test videos: {len(additional_test_fake_videos)} fake, {len(additional_test_real_videos)} real")
    
    # Process additional videos
    print("\nProcessing additional training videos...")
    additional_train_fake_features, additional_train_fake_labels, additional_train_fake_filenames, additional_train_fake_landmarks = process_videos(
        additional_train_fake_videos, 'fake', target_fps=target_fps, max_frames=max_frames
    )
    
    additional_train_real_features, additional_train_real_labels, additional_train_real_filenames, additional_train_real_landmarks = process_videos(
        additional_train_real_videos, 'real', target_fps=target_fps, max_frames=max_frames
    )
    
    print("\nProcessing additional test videos...")
    additional_test_fake_features, additional_test_fake_labels, additional_test_fake_filenames, additional_test_fake_landmarks = process_videos(
        additional_test_fake_videos, 'fake', target_fps=target_fps, max_frames=max_frames
    )
    
    additional_test_real_features, additional_test_real_labels, additional_test_real_filenames, additional_test_real_landmarks = process_videos(
        additional_test_real_videos, 'real', target_fps=target_fps, max_frames=max_frames
    )
    
    # Combine additional landmarks and save them
    additional_landmarks = {}
    additional_landmarks.update(additional_train_fake_landmarks)
    additional_landmarks.update(additional_train_real_landmarks)
    additional_landmarks.update(additional_test_fake_landmarks)
    additional_landmarks.update(additional_test_real_landmarks)
    
    # Save additional landmarks
    print(f"Saving {len(additional_landmarks)} additional landmarks...")
    save_landmarks(additional_landmarks, expanded_landmarks_dir)
    
    # Combine additional training features
    additional_train_features = np.vstack((
        additional_train_fake_features, 
        additional_train_real_features
    )) if len(additional_train_fake_features) > 0 and len(additional_train_real_features) > 0 else np.array([])
    
    additional_train_labels = np.concatenate((
        additional_train_fake_labels, 
        additional_train_real_labels
    )) if len(additional_train_fake_labels) > 0 and len(additional_train_real_labels) > 0 else np.array([])
    
    additional_train_filenames = additional_train_fake_filenames + additional_train_real_filenames
    
    # Combine additional test features
    additional_test_features = np.vstack((
        additional_test_fake_features, 
        additional_test_real_features
    )) if len(additional_test_fake_features) > 0 and len(additional_test_real_features) > 0 else np.array([])
    
    additional_test_labels = np.concatenate((
        additional_test_fake_labels, 
        additional_test_real_labels
    )) if len(additional_test_fake_labels) > 0 and len(additional_test_real_labels) > 0 else np.array([])
    
    additional_test_filenames = additional_test_fake_filenames + additional_test_real_filenames
    
    # Save additional features
    if len(additional_train_features) > 0:
        print(f"Saving {len(additional_train_features)} additional training features...")
        save_features_to_csv(
            additional_train_features, 
            additional_train_labels, 
            additional_train_filenames, 
            os.path.join(expanded_features_dir, "additional_train_features.csv"),
            feature_names
        )
    else:
        print("No additional training features to save.")
    
    if len(additional_test_features) > 0:
        print(f"Saving {len(additional_test_features)} additional test features...")
        save_features_to_csv(
            additional_test_features, 
            additional_test_labels, 
            additional_test_filenames, 
            os.path.join(expanded_features_dir, "additional_test_features.csv"),
            feature_names
        )
    else:
        print("No additional test features to save.")
    
    # Now combine existing and additional features for the expanded dataset
    expanded_train_features = np.vstack((existing_train_features, additional_train_features)) if len(additional_train_features) > 0 else existing_train_features
    expanded_train_labels = np.concatenate((existing_train_labels, additional_train_labels)) if len(additional_train_labels) > 0 else existing_train_labels
    expanded_train_filenames = existing_train_filenames + additional_train_filenames
    
    expanded_test_features = np.vstack((existing_test_features, additional_test_features)) if len(additional_test_features) > 0 else existing_test_features
    expanded_test_labels = np.concatenate((existing_test_labels, additional_test_labels)) if len(additional_test_labels) > 0 else existing_test_labels
    expanded_test_filenames = existing_test_filenames + additional_test_filenames
    
    # Save expanded features
    print("\nSaving expanded dataset features...")
    save_features_to_csv(
        expanded_train_features,
        expanded_train_labels,
        expanded_train_filenames,
        os.path.join(expanded_features_dir, "expanded_train_features.csv"),
        feature_names
    )
    
    save_features_to_csv(
        expanded_test_features,
        expanded_test_labels,
        expanded_test_filenames,
        os.path.join(expanded_features_dir, "expanded_test_features.csv"),
        feature_names
    )
    
    # All features combined
    expanded_all_features = np.vstack((expanded_train_features, expanded_test_features))
    expanded_all_labels = np.concatenate((expanded_train_labels, expanded_test_labels))
    expanded_all_filenames = expanded_train_filenames + expanded_test_filenames
    
    save_features_to_csv(
        expanded_all_features,
        expanded_all_labels,
        expanded_all_filenames,
        os.path.join(expanded_features_dir, "expanded_all_features.csv"),
        feature_names
    )
    
    print(f"\nExpanded dataset summary:")
    print(f"Training set: {len(expanded_train_features)} videos ({len(existing_train_features)} original + {len(additional_train_features)} new)")
    print(f"Test set: {len(expanded_test_features)} videos ({len(existing_test_features)} original + {len(additional_test_features)} new)")
    print(f"Total: {len(expanded_all_features)} videos")
    
    # Generate visualizations for expanded dataset
    print("\nGenerating visualizations for expanded dataset...")
    visualize_tsne(expanded_all_features, expanded_all_labels, expanded_all_filenames, 'expanded_tsne_visualization.png')
    visualize_pca(expanded_all_features, expanded_all_labels, expanded_all_filenames, 'expanded_pca_visualization.png')
    
    # Training and evaluating model on expanded dataset
    print("\nTraining and evaluating model on expanded dataset...")
    expanded_clf = train_and_evaluate_model(expanded_train_features, expanded_train_labels, expanded_test_features, expanded_test_labels)
    
    # Analyze prediction errors
    expanded_predictions = expanded_clf.predict(expanded_test_features)
    analyze_prediction_errors(expanded_test_features, expanded_test_labels, expanded_predictions, expanded_test_filenames)
    
    print("\nAnalysis complete! Results saved in the expanded_features directory.")

if __name__ == "__main__":
    main()