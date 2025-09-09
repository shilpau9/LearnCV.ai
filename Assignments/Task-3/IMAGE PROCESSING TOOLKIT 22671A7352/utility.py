"""
utils.py - Utility Functions for Image Processing GUI
====================================================

This module contains helper functions and utilities used by the main application.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from PIL import Image
import io
import base64

# =============================================================================
# IMAGE PROCESSING UTILITIES
# =============================================================================

def validate_image(image):
    """Validate image format and properties"""
    if image is None:
        raise ValueError("Image is None")
    
    if len(image.shape) not in [2, 3]:
        raise ValueError("Image must be 2D (grayscale) or 3D (color)")
    
    if image.dtype not in [np.uint8, np.float32, np.float64]:
        raise ValueError("Unsupported image data type")
    
    return True

def normalize_image(image):
    """Normalize image to 0-255 range"""
    if image.dtype == np.float32 or image.dtype == np.float64:
        image = np.clip(image, 0, 1)
        return (image * 255).astype(np.uint8)
    return image

def resize_image(image, max_size=800):
    """Resize image while maintaining aspect ratio"""
    h, w = image.shape[:2]
    if max(h, w) > max_size:
        scale = max_size / max(h, w)
        new_w = int(w * scale)
        new_h = int(h * scale)
        return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)
    return image

def calculate_image_stats(image):
    """Calculate comprehensive image statistics"""
    stats = {
        'shape': image.shape,
        'dtype': str(image.dtype),
        'size': image.size,
        'mean': float(np.mean(image)),
        'std': float(np.std(image)),
        'min': int(np.min(image)),
        'max': int(np.max(image)),
        'memory_mb': image.nbytes / (1024 * 1024)
    }
    
    if len(image.shape) == 3:
        stats['channels'] = image.shape[2]
        for i, channel in enumerate(['Blue', 'Green', 'Red']):
            stats[f'{channel.lower()}_mean'] = float(np.mean(image[:,:,i]))
            stats[f'{channel.lower()}_std'] = float(np.std(image[:,:,i]))
    
    return stats

# =============================================================================
# QUALITY ASSESSMENT METRICS
# =============================================================================

def calculate_psnr(original, processed):
    """Calculate Peak Signal-to-Noise Ratio"""
    mse = np.mean((original.astype(float) - processed.astype(float)) ** 2)
    if mse == 0:
        return float('inf')
    
    max_pixel = 255.0
    psnr = 20 * np.log10(max_pixel / np.sqrt(mse))
    return psnr

def calculate_ssim(img1, img2):
    """Calculate Structural Similarity Index (simplified version)"""
    # Convert to float
    img1 = img1.astype(float)
    img2 = img2.astype(float)
    
    # Calculate means
    mu1 = np.mean(img1)
    mu2 = np.mean(img2)
    
    # Calculate variances and covariance
    var1 = np.var(img1)
    var2 = np.var(img2)
    covar = np.mean((img1 - mu1) * (img2 - mu2))
    
    # SSIM constants
    c1 = (0.01 * 255) ** 2
    c2 = (0.03 * 255) ** 2
    
    # Calculate SSIM
    ssim = ((2 * mu1 * mu2 + c1) * (2 * covar + c2)) / \
           ((mu1**2 + mu2**2 + c1) * (var1 + var2 + c2))
    
    return ssim

def calculate_mse(img1, img2):
    """Calculate Mean Squared Error"""
    return np.mean((img1.astype(float) - img2.astype(float)) ** 2)

# =============================================================================
# SPECIALIZED PROCESSING FUNCTIONS
# =============================================================================

def adaptive_threshold_multi(image):
    """Apply multiple adaptive thresholding methods"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    methods = {
        'adaptive_mean': cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                             cv2.THRESH_BINARY, 11, 2),
        'adaptive_gaussian': cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                                 cv2.THRESH_BINARY, 11, 2),
        'otsu': cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    }
    
    return methods

def noise_analysis(image):
    """Analyze different types of noise in image"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Estimate noise using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    
    # Calculate image gradients for texture analysis
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    
    analysis = {
        'laplacian_variance': laplacian_var,
        'gradient_mean': np.mean(gradient_magnitude),
        'gradient_std': np.std(gradient_magnitude),
        'noise_level': 'Low' if laplacian_var > 100 else 'High' if laplacian_var < 30 else 'Medium'
    }
    
    return analysis

def create_custom_kernel(kernel_type, size=3):
    """Create custom convolution kernels"""
    kernels = {
        'sharpen': np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]]),
        'edge_enhance': np.array([[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]]),
        'emboss': np.array([[-2, -1, 0], [-1, 1, 1], [0, 1, 2]]),
        'blur': np.ones((size, size)) / (size * size),
        'gaussian': cv2.getGaussianKernel(size, size/3)
    }
    
    if kernel_type in kernels:
        return kernels[kernel_type]
    else:
        return np.ones((3, 3)) / 9  # Default blur kernel

# =============================================================================
# IMAGE ENHANCEMENT UTILITIES
# =============================================================================

def auto_enhance(image):
    """Automatic image enhancement pipeline"""
    # Convert to grayscale for analysis
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image
    
    # Analyze histogram
    hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
    
    # Determine enhancement strategy based on histogram
    mean_intensity = np.mean(gray)
    std_intensity = np.std(gray)
    
    enhanced = image.copy()
    
    # Apply enhancement based on image characteristics
    if mean_intensity < 100:  # Dark image
        # Apply gamma correction for brightening
        gamma = 1.5
        enhanced = np.power(enhanced / 255.0, gamma) * 255.0
        enhanced = enhanced.astype(np.uint8)
    elif std_intensity < 50:  # Low contrast
        # Apply CLAHE for contrast enhancement
        if len(enhanced.shape) == 3:
            yuv = cv2.cvtColor(enhanced, cv2.COLOR_BGR2YUV)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            yuv[:,:,0] = clahe.apply(yuv[:,:,0])
            enhanced = cv2.cvtColor(yuv, cv2.COLOR_YUV2BGR)
        else:
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8,8))
            enhanced = clahe.apply(enhanced)
    
    return enhanced

def create_artistic_effect(image, effect_type='oil_painting'):
    """Create artistic effects"""
    if effect_type == 'oil_painting':
        # Oil painting effect using bilateral filter and edge enhancement
        smooth = cv2.bilateralFilter(image, 15, 80, 80)
        smooth = cv2.bilateralFilter(smooth, 15, 80, 80)  # Apply twice for stronger effect
        return smooth
    
    elif effect_type == 'pencil_sketch':
        # Pencil sketch effect
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 7, 7)
        return edges
    
    elif effect_type == 'cartoon':
        # Cartoon effect
        # Bilateral filter to reduce noise while keeping edges sharp
        bilateral = cv2.bilateralFilter(image, 15, 80, 80)
        
        # Convert to grayscale and apply median blur
        gray = cv2.cvtColor(bilateral, cv2.COLOR_BGR2GRAY)
        gray_blur = cv2.medianBlur(gray, 5)
        
        # Create edge mask
        edges = cv2.adaptiveThreshold(gray_blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                    cv2.THRESH_BINARY, 7, 7)
        
        # Convert edges to 3-channel
        edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Bitwise AND to combine
        cartoon = cv2.bitwise_and(bilateral, edges)
        return cartoon
    
    return image

# =============================================================================
# BATCH PROCESSING UTILITIES
# =============================================================================

def batch_process_images(image_list, operation_func, **kwargs):
    """Apply operation to multiple images"""
    processed_images = []
    processing_times = []
    
    import time
    
    for img in image_list:
        start_time = time.time()
        processed = operation_func(img, **kwargs)
        end_time = time.time()
        
        processed_images.append(processed)
        processing_times.append(end_time - start_time)
    
    return processed_images, processing_times

def create_image_montage(images, titles=None, grid_size=None):
    """Create a montage of multiple images"""
    if not images:
        return None
    
    n_images = len(images)
    
    # Determine grid size if not provided
    if grid_size is None:
        cols = int(np.ceil(np.sqrt(n_images)))
        rows = int(np.ceil(n_images / cols))
    else:
        rows, cols = grid_size
    
    # Ensure all images have the same size
    h, w = images[0].shape[:2]
    resized_images = []
    
    for img in images:
        if img.shape[:2] != (h, w):
            img_resized = cv2.resize(img, (w, h))
        else:
            img_resized = img
        
        # Convert grayscale to BGR for consistency
        if len(img_resized.shape) == 2:
            img_resized = cv2.cvtColor(img_resized, cv2.COLOR_GRAY2BGR)
        
        resized_images.append(img_resized)
    
    # Create montage
    montage_rows = []
    
    for row in range(rows):
        row_images = []
        for col in range(cols):
            idx = row * cols + col
            if idx < len(resized_images):
                img = resized_images[idx]
                
                # Add title if provided
                if titles and idx < len(titles):
                    # Add text to image
                    img_with_title = img.copy()
                    cv2.putText(img_with_title, titles[idx], (10, 30), 
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                    row_images.append(img_with_title)
                else:
                    row_images.append(img)
            else:
                # Fill empty spaces with black image
                empty_img = np.zeros((h, w, 3), dtype=np.uint8)
                row_images.append(empty_img)
        
        # Concatenate images horizontally for this row
        if row_images:
            montage_rows.append(np.hstack(row_images))
    
    # Concatenate rows vertically
    if montage_rows:
        montage = np.vstack(montage_rows)
        return montage
    
    return None

# =============================================================================
# FILE I/O UTILITIES
# =============================================================================

def save_image_with_metadata(image, filename, metadata=None):
    """Save image with