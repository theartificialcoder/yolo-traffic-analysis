# ==============================================================================
#           YOLO Dataset Analysis Script: Full Unified Image Analysis
#
# Description:
#   This script analyzes EVERY image in the COMBINED train and validation sets.
#   It uses a caching system to perform the slow analysis only once.
#   (Version 1.2 - Replaced dimension histograms with a scatter plot)
# ==============================================================================

import os
import cv2
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION ---
TRAIN_DIR = '../datasets/train'
VAL_DIR = '../datasets/val'
REPORT_DIR = '../report/image_full_unified'
CACHE_FILE = 'unified_image_data.pkl' # Cache file to save parsed data

# --- 2. HELPER FUNCTION TO GET ALL IMAGE PATHS ---
def get_all_image_paths(data_dir):
    """Scans a directory and returns a list of full paths to all image files."""
    if not os.path.isdir(data_dir):
        print(f"Warning: Directory not found at '{data_dir}'. Skipping.")
        return []
    return [os.path.join(data_dir, f) for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]

# --- 3. PLOTTING FUNCTIONS ---

def plot_resolution_scatterplot(df, title, filename):
    """
    Generates a single scatter plot of all unique image resolutions.
    This replaces the separate width and height histograms.
    """
    print(f"  - Generating plot: {title}")
    # Get unique width/height pairs to avoid overplotting and make the chart cleaner
    unique_resolutions = df[['img_width', 'img_height']].drop_duplicates()
    
    plt.figure(figsize=(10, 8))
    sns.scatterplot(
        x='img_width',
        y='img_height',
        data=unique_resolutions,
        alpha=0.6,      # Use transparency to show areas of high density
        s=40            # Set a consistent size for the points
    )
    plt.title(title, fontsize=16)
    plt.xlabel('Image Width (pixels)', fontsize=12)
    plt.ylabel('Image Height (pixels)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_brightness(df, title, filename):
    """Plots the brightness distribution for the full dataset."""
    print(f"  - Generating plot: {title}")
    plt.figure(figsize=(10, 6))
    sns.histplot(data=df, x='brightness', bins=50, kde=True, color='orange')
    plt.title(title, fontsize=16)
    plt.xlabel('Average Pixel Intensity (0=Black, 255=White)')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_bbox_sizes(df, title, filename):
    """Plots a histogram of bounding box areas in pixels for the full dataset."""
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

# --- 4. MAIN EXECUTION ---
def main():
    """Main function to orchestrate the full unified image analysis."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"Full image analysis reports will be saved to '{REPORT_DIR}'")

    class_names = [
        'Motorcycle', 'Car', 'Bus', 'Truck', 'Transporter',
        'Container', 'Big_Transporter'
    ]

    # --- Caching Logic ---
    if os.path.exists(CACHE_FILE):
        print(f"\n✅ Found cache file '{CACHE_FILE}'. Loading pre-parsed data...")
        cached_data = pd.read_pickle(CACHE_FILE)
        bbox_df = cached_data['bbox_df']
        image_df = cached_data['image_df']
        print("🎉 Data loaded instantly from cache.")
    else:
        print(f"\nℹ️ No cache file found. Starting full dataset analysis (this will take a long time)...")
        
        train_image_paths = get_all_image_paths(TRAIN_DIR)
        val_image_paths = get_all_image_paths(VAL_DIR)
        all_image_paths = train_image_paths + val_image_paths

        if not all_image_paths:
            print("\nCritical Error: No images found in any directory. Exiting.")
            return

        print(f"\n--- Analyzing all {len(all_image_paths)} images from the full dataset... ---")
        bbox_data = []
        image_data = []
        for i, img_path in enumerate(all_image_paths):
            if (i + 1) % 100 == 0 or i == len(all_image_paths) - 1:
                print(f"  - Processing image [{i + 1}/{len(all_image_paths)}]")

            label_path = os.path.splitext(img_path)[0] + '.txt'

            image = cv2.imread(img_path)
            if image is None: continue
            height, width, _ = image.shape
            brightness = np.mean(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
            image_data.append([width, height, brightness])

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
        image_df = pd.DataFrame(image_data, columns=['img_width', 'img_height', 'brightness'])

        print(f"\n💾 Saving parsed data to '{CACHE_FILE}' for future runs...")
        pd.to_pickle({'bbox_df': bbox_df, 'image_df': image_df}, CACHE_FILE)
        print("✅ Data saved.")

    if bbox_df.empty or image_df.empty:
        print("\nCritical Error: No data was parsed. Exiting.")
        return

    # --- Generate Plots ---
    print("\n--- Generating Plots from the Full Unified Dataset ---")
    # UPDATED: Call the new scatter plot function for image dimensions
    plot_resolution_scatterplot(image_df, 'Image Resolution Distribution', '1_image_resolutions.png')
    plot_brightness(image_df, 'Brightness Distribution', '2_brightness.png')
    plot_bbox_sizes(bbox_df, 'Bounding Box Sizes', '3_bbox_sizes.png')

    print("\n✅ Full unified image analysis complete!")

if __name__ == '__main__':
    main()