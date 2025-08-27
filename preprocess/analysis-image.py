# ==============================================================================
#           YOLO Dataset Analysis Script 2: Image & Label Sample
#
# Description:
#   This script analyzes a random sample of images and their corresponding
#   labels. It's much faster than analyzing the full dataset and provides a
#   statistically representative view of image properties (dimensions, brightness)
#   and absolute object sizes (in pixels).
#
#   UPDATED: Includes a single, combined plot for image resolutions.
#
# Instructions:
#   1. Requires OpenCV: pip install opencv-python
#   2. Place this script in the same parent directory as your 'datasets' folder.
#   3. Ensure the CONFIGURATION variables below are correct.
#   4. Run from the terminal: python analyze_images_sample.py
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
VAL_DIR = '../datasets/val'
REPORT_DIR = '../report/image'
SAMPLE_SIZE = 5000  # Number of random images to analyze
CLASS_NAMES = [
    'Motorcycle', 'Car', 'Bus', 'Truck', 'Transporter',
    'Container', 'Big Transporter'
]

# --- 2. CORE PARSING FUNCTION ---
def parse_image_sample(data_dir, class_names, sample_size, set_name):
    """
    Parses a random sample of images and their labels.
    Returns two DataFrames: one for bounding box data, one for unique image data.
    """
    bbox_data = []
    image_data = [] # To store unique image properties

    if not os.path.isdir(data_dir):
        print(f"Error: Directory not found at '{data_dir}'.")
        return pd.DataFrame(), pd.DataFrame()

    all_image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    sample_files = random.sample(all_image_files, min(len(all_image_files), sample_size))
    print(f"Found {len(all_image_files)} total images in '{set_name}' set. Analyzing a random sample of {len(sample_files)}...")

    for i, img_file in enumerate(sample_files):
        if (i + 1) % 50 == 0 or i == len(sample_files) - 1:
            print(f"  - Processing image [{i + 1}/{len(sample_files)}]")

        img_path = os.path.join(data_dir, img_file)
        label_path = os.path.join(data_dir, os.path.splitext(img_file)[0] + '.txt')

        image = cv2.imread(img_path)
        if image is None: continue
        height, width, _ = image.shape
        brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))

        # Store unique image data once per image
        image_data.append([width, height, brightness, set_name])

        # Read label properties for bbox data
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                for line in f:
                    if line.strip():
                        parts = line.strip().split()
                        if len(parts) < 5: continue
                        class_id = int(parts[0])
                        w, h = float(parts[3]), float(parts[4])
                        bbox_data.append([class_names[class_id], w * width, h * height])

    bbox_df = pd.DataFrame(bbox_data, columns=['class_name', 'bbox_width_px', 'bbox_height_px'])
    image_df = pd.DataFrame(image_data, columns=['img_width', 'img_height', 'brightness', 'dataset'])
    
    return bbox_df, image_df

# --- 3. PLOTTING FUNCTIONS ---
def plot_brightness(df, title, filename):
    """Plots the brightness distribution to check for day/night bias."""
    print(f"  - Generating plot: {title}")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='brightness', bins=50, kde=True, color='orange')
    plt.title(title, fontsize=16)
    plt.xlabel('Average Pixel Intensity (0=Black, 255=White)')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_bbox_sizes(df, title, filename):
    """Plots a histogram of bounding box areas in pixels."""
    print(f"  - Generating plot: {title}")
    df['area_px'] = df['bbox_width_px'] * df['bbox_height_px']
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='area_px', bins=50, kde=True)
    plt.title(title, fontsize=16)
    plt.xlabel('Bounding Box Area (pixels^2)')
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()
    
def plot_resolution_scatterplot(df, title, filename):
    """Generates a single scatter plot of image width vs. height."""
    print(f"  - Generating plot: {title}")
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='img_width',
        y='img_height',
        hue='dataset',
        data=df,
        alpha=0.6,
        s=50
    )
    plt.title(title, fontsize=16)
    plt.xlabel('Image Width (pixels)', fontsize=12)
    plt.ylabel('Image Height (pixels)', fontsize=12)
    plt.grid(True)
    plt.legend(title='Dataset')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

# --- 4. MAIN EXECUTION ---
def main():
    """Main function to orchestrate the image sample analysis."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"Image sample analysis reports will be saved to '{REPORT_DIR}'")
    
    # --- Parse Data from Samples ---
    train_bbox_df, train_image_df = parse_image_sample(TRAIN_DIR, CLASS_NAMES, SAMPLE_SIZE, 'train')
    val_bbox_df, val_image_df = parse_image_sample(VAL_DIR, CLASS_NAMES, SAMPLE_SIZE, 'val')
    
    if train_bbox_df.empty or val_bbox_df.empty:
        print("\nCritical Error: Could not parse sample files. Exiting.")
        return

    # --- Generate Plots ---
    print("\n--- Generating Plots from Sample Data ---")
    
    # Bbox plots
    plot_bbox_sizes(train_bbox_df, 'Absolute Bounding Box Sizes (Training Sample)', '1_bbox_sizes_train.png')
    plot_bbox_sizes(val_bbox_df, 'Absolute Bounding Box Sizes (Validation Sample)', '1_bbox_sizes_val.png')

    # Brightness plots
    plot_brightness(train_image_df, 'Brightness Distribution (Training Sample)', '2_brightness_train.png')
    plot_brightness(val_image_df, 'Brightness Distribution (Validation Sample)', '2_brightness_val.png')
    
    # NEW: Combined Resolution Plot
    combined_image_df = pd.concat([train_image_df, val_image_df]).drop_duplicates()
    plot_resolution_scatterplot(combined_image_df, 'Distribution of Image Resolutions (Sample)', '3_image_resolutions.png')
    
    print("\n✅ Image sample analysis complete!")

if __name__ == '__main__':
    main()