#!/usr/bin/env python
"""
Extract transmission spectrum data from a plot image.
This script digitizes the transmission.png plot and saves the values to an ASCII file.
"""

import numpy as np
import cv2
from scipy.ndimage import median_filter
import os


def extract_spectrum_from_image(image_path, output_path):
    """
    Extract spectrum data from a plot image.
    
    Parameters:
    -----------
    image_path : str
        Path to the input image
    output_path : str
        Path to save the ASCII output file
    """
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    
    # Get image dimensions
    height, width = img.shape[:2]
    print(f"Image size: {width} x {height}")
    
    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Define the plot area (excluding axis labels)
    # These margins work for typical matplotlib plots
    plot_left = int(width * 0.08)
    plot_right = int(width * 0.98)
    plot_top = int(height * 0.05)
    plot_bottom = int(height * 0.85)
    
    # Extract the plot region
    plot_gray = gray[plot_top:plot_bottom, plot_left:plot_right]
    plot_height, plot_width = plot_gray.shape
    
    print(f"Plot area: {plot_width} x {plot_height}")
    
    # Method: For each column, find the darkest pixel(s)
    # Handle multiple lines by tracking continuity
    x_values = []
    y_values = []
    
    for x in range(plot_width):
        column = plot_gray[:, x]
        
        # Find the darkest point(s)
        min_val = np.min(column)
        
        # Only consider if there's a clear dark feature (line drawn)
        if min_val < 150:
            # Find all pixels within some threshold of the minimum
            dark_mask = column < min_val + 20
            dark_pixels = np.where(dark_mask)[0]
            
            if len(dark_pixels) > 0:
                # Check if there are multiple distinct clusters (multiple lines)
                if len(dark_pixels) > 1:
                    diffs = np.diff(dark_pixels)
                    gaps = np.where(diffs > 10)[0]
                    
                    if len(gaps) > 0:
                        # Multiple clusters - split and choose best one
                        clusters = np.split(dark_pixels, gaps + 1)
                        
                        if len(y_values) > 0:
                            # Choose cluster closest to last point (continuity)
                            last_y = y_values[-1]
                            best_cluster = min(clusters, key=lambda c: abs(np.mean(c) - last_y))
                        else:
                            # Start with upper cluster (smaller y = higher position)
                            best_cluster = min(clusters, key=lambda c: np.mean(c))
                        
                        y_pos = np.mean(best_cluster)
                    else:
                        y_pos = np.mean(dark_pixels)
                else:
                    y_pos = dark_pixels[0]
                
                x_values.append(x)
                y_values.append(y_pos)
    
    x_values = np.array(x_values)
    y_values = np.array(y_values)
    
    print(f"Extracted {len(x_values)} raw points")
    
    # Apply smoothing to remove noise - but be less aggressive
    if len(y_values) > 20:
        y_smooth = median_filter(y_values, size=5)
        
        # Remove extreme outliers only
        diff = np.abs(y_values - y_smooth)
        median_diff = np.median(diff)
        if median_diff > 0:
            threshold = 10 * median_diff  # More permissive
            good = diff < threshold
            
            # Only apply if we keep enough points
            if np.sum(good) > len(x_values) * 0.5:
                x_values = x_values[good]
                y_values = y_values[good]
                print(f"After cleaning: {len(x_values)} points")
    
    if len(x_values) == 0:
        raise ValueError("No valid data points extracted from image")
    
    # Normalize coordinates
    x_normalized = x_values / plot_width
    # Invert y (image y=0 is top, but we want transmission increasing upward)
    y_normalized = 1.0 - (y_values / plot_height)
    
    print(f"Value ranges:")
    print(f"  X (normalized): {x_normalized.min():.4f} to {x_normalized.max():.4f}")
    print(f"  Y (normalized): {y_normalized.min():.4f} to {y_normalized.max():.4f}")
    
    # Save to ASCII file
    header = """# Transmission spectrum extracted from transmission.png
# Columns: pixel_x, pixel_y, x_normalized, y_normalized
# Note: x_normalized and y_normalized are in range [0, 1]
# The actual wavelength/frequency and transmission values depend on the original plot axes
# Modify the ranges in the script if you know the actual axis values
#
# pixel_x    pixel_y    x_normalized    y_normalized(transmission)
"""
    
    with open(output_path, 'w') as f:
        f.write(header)
        for i in range(len(x_values)):
            f.write(f"{x_values[i]:8.1f}  {y_values[i]:8.1f}  {x_normalized[i]:.6f}  {y_normalized[i]:.6f}\n")
    
    print(f"\nData saved to: {output_path}")
    print(f"Extracted {len(x_values)} data points")
    
    return x_normalized, y_normalized


if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    image_path = os.path.join(script_dir, "transmission.png")
    output_path = os.path.join(script_dir, "transmission.txt")
    
    x, y = extract_spectrum_from_image(image_path, output_path)
