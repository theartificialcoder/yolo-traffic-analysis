# ==============================================================================
#           YOLO Dataset Analysis Script 3: Correlation Analysis
#
# Description:
#   This script generates plots that show the relationships between different
#   properties of your dataset, such as bounding box sizes for each class
#   and object density versus image brightness.
#
# Instructions:
#   1. Place this script in the same parent directory as your 'datasets' folder.
#   2. Ensure the CONFIGURATION variables are correct.
#   3. Run from the terminal: python analyze_correlations.py
#
# ==============================================================================

import os
import cv2
import random
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION ---
TRAIN_DIR = '../datasets/train'
REPORT_DIR = '../report/correlation'
SAMPLE_SIZE = 5000  # A 5000-image sample is great for these plots
CLASS_NAMES = [
    'Motorcycle', 'Car', 'Bus', 'Truck', 'Transporter',
    'Container', 'Big_Transporter'
]

# --- 2. CORE PARSING FUNCTION ---
def parse_image_sample_for_correlation(data_dir, class_names, sample_size):
    """Parses a random sample, collecting all necessary data for correlation plots."""
    data = []
    all_image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    sample_files = random.sample(all_image_files, min(len(all_image_files), sample_size))
    print(f"Analyzing a random sample of {len(sample_files)} images...")

    for i, img_file in enumerate(sample_files):
        if (i + 1) % 50 == 0:
            print(f"  - Processing image [{i + 1}/{len(sample_files)}]")

        img_path = os.path.join(data_dir, img_file)
        label_path = os.path.join(data_dir, os.path.splitext(img_file)[0] + '.txt')

        image = cv2.imread(img_path)
        if image is None: continue
        height, width, _ = image.shape
        brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        class_id = int(parts[0])
                        w, h = float(parts[3]), float(parts[4])
                        data.append([img_path, width, height, brightness, class_names[class_id], w * width, h * height])

    return pd.DataFrame(data, columns=['image_path', 'img_width', 'img_height', 'brightness', 'class_name', 'bbox_width_px', 'bbox_height_px'])

# --- 3. PLOTTING FUNCTIONS ---
def plot_bbox_area_per_class(df, title, filename):
    """Generates a box plot to compare bounding box areas across classes."""
    print(f"  - Generating plot: {title}")
    df['area_px'] = df['bbox_width_px'] * df['bbox_height_px']
    
    plt.figure(figsize=(14, 8))
    sns.boxplot(x='class_name', y='area_px', data=df, palette='muted', order=sorted(df['class_name'].unique()))
    plt.title(title, fontsize=16)
    plt.xlabel('Class Name', fontsize=12)
    plt.ylabel('Bounding Box Area (pixels^2)', fontsize=12)
    plt.yscale('log') # Log scale is essential for viewing skewed size distributions
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_aspect_ratio_per_class(df, title, filename):
    """Generates a violin plot to compare bounding box shapes across classes."""
    print(f"  - Generating plot: {title}")
    # Avoid division by zero for any zero-height boxes
    df['aspect_ratio'] = df['bbox_width_px'] / (df['bbox_height_px'] + 1e-6)
    
    plt.figure(figsize=(14, 8))
    sns.violinplot(x='class_name', y='aspect_ratio', data=df, palette='husl', order=sorted(df['class_name'].unique()))
    # Add a horizontal line at 1.0 for reference (square shape)
    plt.axhline(1.0, color='red', linestyle='--', label='Square (Width = Height)')
    plt.title(title, fontsize=16)
    plt.xlabel('Class Name', fontsize=12)
    plt.ylabel('Aspect Ratio (Width / Height)', fontsize=12)
    plt.yscale('log') # Log scale helps visualize both wide and tall objects clearly
    plt.xticks(rotation=45, ha='right')
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_objects_vs_brightness(df, title, filename):
    """Generates a 2D histogram (hexbin) of object count vs. image brightness."""
    print(f"  - Generating plot: {title}")
    # First, calculate the number of objects for each unique image
    image_stats = df.groupby('image_path').agg(
        object_count=('class_name', 'size'),
        brightness=('brightness', 'first')
    ).reset_index()

    plt.figure(figsize=(12, 8))
    # A hexbin plot is excellent for showing density in large datasets
    plt.hexbin(x='brightness', y='object_count', data=image_stats, gridsize=30, cmap='inferno')
    plt.colorbar(label='Number of Images in Bin')
    plt.title(title, fontsize=16)
    plt.xlabel('Average Image Brightness (0=Dark, 255=Bright)', fontsize=12)
    plt.ylabel('Number of Objects per Image', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

# --- 4. MAIN EXECUTION ---
def main():
    """Main function to orchestrate the correlation analysis."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"Correlation reports will be saved to '{REPORT_DIR}'")

    # --- Parse Data from a Sample ---
    # We only analyze the training set here, as it's the largest and most representative
    df = parse_image_sample_for_correlation(TRAIN_DIR, CLASS_NAMES, SAMPLE_SIZE)
    if df.empty:
        print("\nCritical Error: Could not parse sample files. Exiting.")
        return
        
    # --- Generate Plots ---
    print("\n--- Generating Correlation Plots from Training Set Sample ---")
    plot_bbox_area_per_class(df, 'Bounding Box Area vs. Class', '1_area_vs_class.png')
    plot_aspect_ratio_per_class(df, 'Bounding Box Aspect Ratio vs. Class', '2_aspect_ratio_vs_class.png')
    plot_objects_vs_brightness(df, 'Object Count vs. Image Brightness', '3_objects_vs_brightness.png')

    print("\n✅ Correlation analysis complete!")

if __name__ == '__main__':
    main()