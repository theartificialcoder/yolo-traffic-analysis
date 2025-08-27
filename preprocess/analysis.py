# ==============================================================================
#           YOLO Dataset Analysis Script: Unified & Consolidated
#
# Description:
#   This script performs a comprehensive analysis of the entire dataset by
#   combining the training and validation sets. It generates a single, unified
#   report with consolidated plots for the most accurate overview.
#
#   Features:
#   - Caching: Parses the full dataset once and saves the result for instant
#     re-analysis on subsequent runs.
#   - Unified Plots: Generates one plot per category for the entire dataset.
#   - Consolidated Visualizations: Combines related metrics (like width/height
#     and brightness/contrast) into single, more insightful charts.
#
# ==============================================================================

import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION ---
TRAIN_DIR = '../datasets/train'
VAL_DIR = '../datasets/val'
REPORT_DIR = '../report'
CACHE_FILE = 'unified_dataset.pkl'  # File to store the parsed DataFrame
CLASS_NAMES = [
    'Motorcycle', 'Car', 'Bus', 'Truck', 'Transporter',
    'Container', 'Big_Transporter'
]

# --- 2. CORE PARSING FUNCTION ---
def parse_yolo_data(data_dir, class_names):
    """Parses all image and label files in a directory."""
    data = []
    if not os.path.isdir(data_dir):
        print(f"❌ Error: Directory not found at '{data_dir}'.")
        return pd.DataFrame()

    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)
    print(f"✅ Found {total_images} images in '{data_dir}'. Starting processing...")

    for i, img_file in enumerate(image_files):
        if (i + 1) % 100 == 0 or i == total_images - 1:
            print(f"  ➡️ Processing image [{i + 1}/{total_images}]")

        img_path = os.path.join(data_dir, img_file)
        label_path = os.path.join(data_dir, os.path.splitext(img_file)[0] + '.txt')

        image = cv2.imread(img_path)
        if image is None: continue
        height, width, _ = image.shape
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        brightness = np.mean(gray_image)
        contrast = np.std(gray_image)

        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                if not lines:
                    data.append([img_path, width, height, brightness, contrast, -1, 'no_object', 0, 0, 0, 0])
                else:
                    for line in f:
                        if line.strip():
                            parts = line.strip().split()
                            if len(parts) < 5: continue
                            class_id = int(parts[0])
                            x_c, y_c, w, h = map(float, parts[1:])
                            data.append([img_path, width, height, brightness, contrast, class_id, class_names[class_id], x_c, y_c, w, h])
        else:
            data.append([img_path, width, height, brightness, contrast, -1, 'no_object', 0, 0, 0, 0])

    columns = ['image_path', 'img_width', 'img_height', 'brightness', 'contrast', 'class_id', 'class_name', 'x_center', 'y_center', 'width', 'height']
    return pd.DataFrame(data, columns=columns)

# --- 3. PLOTTING FUNCTIONS ---
# All plotting functions now accept a single DataFrame and save one file.

def plot_class_distribution(df, filename):
    """Generates a bar chart of instance counts for each class."""
    print(f"  - Generating Class Distribution plot...")
    plt.figure(figsize=(12, 7))
    class_counts = df[df['class_name'] != 'no_object']['class_name'].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis', order=class_counts.index)
    plt.title('Class Distribution (Full Dataset)', fontsize=16)
    plt.ylabel('Number of Instances (Bounding Boxes)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_image_resolutions(df, filename):
    """Generates a single scatter plot of all unique image resolutions."""
    print(f"  - Generating Image Resolutions plot...")
    unique_resolutions = df[['img_width', 'img_height']].drop_duplicates()
    plt.figure(figsize=(10, 8))
    sns.scatterplot(x='img_width', y='img_height', data=unique_resolutions, alpha=0.5, s=30)
    plt.title('Distribution of Image Resolutions (Full Dataset)', fontsize=16)
    plt.xlabel('Image Width (pixels)', fontsize=12)
    plt.ylabel('Image Height (pixels)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_brightness_contrast(df, filename):
    """
    Generates a jointplot showing the relationship between image brightness and contrast.
    
    UPDATED: Uses kind='scatter' with alpha transparency, which is more robust
    to datasets containing images with zero contrast, preventing ZeroDivisionError.
    """
    print(f"  - Generating Brightness vs. Contrast plot...")
    unique_images = df[['image_path', 'brightness', 'contrast']].drop_duplicates()

    # --- SAFETY CHECK ---
    # Before plotting, ensure there is more than one unique value to avoid errors.
    if unique_images['brightness'].nunique() <= 1 or unique_images['contrast'].nunique() <= 1:
        print(f"  - ⚠️ Skipping Brightness vs. Contrast plot: Not enough variance in data.")
        return

    # Use a scatter plot which is robust to zero-variance data, with alpha for density.
    g = sns.jointplot(
        data=unique_images, 
        x='brightness', 
        y='contrast', 
        kind='scatter', # Changed from 'hex' to 'scatter'
        height=8,
        joint_kws={'alpha': 0.2, 's': 20} # Add transparency and size to scatter points
    )
    
    g.fig.suptitle('Brightness vs. Contrast (Full Dataset)', y=1.01)
    g.set_axis_labels('Average Brightness (0-255)', 'Contrast (Std. Dev. of Pixels)')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_bbox_dimensions(df, filename):
    """
    Generates a jointplot of bounding box widths and heights (in pixels).
    This visualization effectively shows common object shapes and sizes.
    """
    print(f"  - Generating Bounding Box Dimensions plot...")
    df_obj = df[df['class_name'] != 'no_object'].copy()
    df_obj['abs_width'] = df_obj['width'] * df_obj['img_width']
    df_obj['abs_height'] = df_obj['height'] * df_obj['img_height']
    # Use a sample for the scatter part of the jointplot to keep it readable
    sample_df = df_obj.sample(min(len(df_obj), 10000))
    g = sns.jointplot(data=sample_df, x='abs_width', y='abs_height', kind='scatter', height=8, alpha=0.3)
    g.fig.suptitle('Bounding Box Dimensions (Full Dataset)', y=1.01)
    g.set_axis_labels('Bounding Box Width (pixels)', 'Bounding Box Height (pixels)')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()
    
# --- 4. MAIN EXECUTION SCRIPT ---
def main():
    """Main function to run the complete, unified analysis."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"Analysis reports will be saved to '{REPORT_DIR}' directory.")
    
    # --- Step 1: Caching Logic ---
    # Check if a cached DataFrame already exists to save time.
    if os.path.exists(CACHE_FILE):
        print(f"\n✅ Found cache file '{CACHE_FILE}'. Loading pre-parsed data...")
        combined_df = pd.read_pickle(CACHE_FILE)
        print("🎉 Data loaded instantly from cache.")
    else:
        print(f"\nℹ️ No cache file found. Starting full dataset parse (this will run only once)...")
        # Parse both train and val sets
        train_df = parse_yolo_data(TRAIN_DIR, CLASS_NAMES)
        val_df = parse_yolo_data(VAL_DIR, CLASS_NAMES)
        
        # Combine them into a single DataFrame
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        
        if combined_df.empty:
            print("\n❌ Critical Error: No data was parsed. Please check dataset paths. Exiting.")
            return
            
        # Save the combined DataFrame to the cache file for future runs
        print(f"\n💾 Saving parsed data to '{CACHE_FILE}' for next time...")
        combined_df.to_pickle(CACHE_FILE)
        print("✅ Data saved.")

    # --- Step 2: High-Level Overview ---
    print("\n" + "="*50)
    print("      UNIFIED DATASET OVERVIEW (TRAIN + VAL)")
    print("="*50)
    print(f"Total Unique Images: {combined_df['image_path'].nunique()}")
    print(f"Total Bounding Boxes: {len(combined_df[combined_df['class_name'] != 'no_object'])}")
    print("="*50)

    # --- Step 3: Generate and Save All Plots ---
    print("\n--- GENERATING UNIFIED VISUALIZATION REPORT ---")
    plot_class_distribution(combined_df, '1_class_distribution.png')
    plot_image_resolutions(combined_df, '2_image_resolutions.png')
    plot_brightness_contrast(combined_df, '3_brightness_vs_contrast.png')
    plot_bbox_dimensions(combined_df, '4_bbox_dimensions.png')
    
    print("\n" + "="*50)
    print("✅ UNIFIED ANALYSIS COMPLETE!")
    print(f"All reports saved to '{REPORT_DIR}'.")
    print("="*50)

if __name__ == '__main__':
    main()