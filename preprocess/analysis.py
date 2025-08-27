# ==============================================================================
#           YOLO Dataset Analysis Script: Unified & Robust Plots
#
# Description:
#   This script analyzes the entire dataset (train + val) and uses simple,
#   robust plotting methods to ensure reliable visualization, even with
#   large or unusual datasets. This version reverts to separate histograms
#   for dimensions and image properties to avoid errors from complex plot types.
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
REPORT_DIR = '../report/unified_analysis_robust'
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

# --- 3. ROBUST PLOTTING FUNCTIONS ---

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

def plot_image_dimensions_separate(df, filename):
    """Generates two separate histograms for image width and height."""
    print(f"  - Generating Image Dimensions plot...")
    dims = df[['image_path', 'img_width', 'img_height']].drop_duplicates()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Image Dimensions (Full Dataset)', fontsize=16)
    
    sns.histplot(dims['img_width'], ax=axes[0], bins=50, kde=True, color='skyblue')
    axes[0].set_title('Image Width Distribution')
    axes[0].set_xlabel('Width (pixels)')
    
    sns.histplot(dims['img_height'], ax=axes[1], bins=50, kde=True, color='salmon')
    axes[1].set_title('Image Height Distribution')
    axes[1].set_xlabel('Height (pixels)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_brightness_contrast_separate(df, filename):
    """Generates two separate histograms for image brightness and contrast."""
    print(f"  - Generating Brightness and Contrast plot...")
    dims = df[['image_path', 'brightness', 'contrast']].drop_duplicates()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Image Brightness & Contrast (Full Dataset)', fontsize=16)
    
    sns.histplot(dims['brightness'], ax=axes[0], bins=50, kde=True, color='orange')
    axes[0].set_title('Brightness Distribution')
    axes[0].set_xlabel('Average Pixel Intensity (0-255)')
    
    sns.histplot(dims['contrast'], ax=axes[1], bins=50, kde=True, color='purple')
    axes[1].set_title('Contrast Distribution (Std Dev)')
    axes[1].set_xlabel('Standard Deviation of Pixel Intensity')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_bbox_size_distribution(df, filename):
    """Generates a histogram of bounding box areas."""
    print(f"  - Generating Bounding Box Size plot...")
    df_obj = df[df['class_name'] != 'no_object'].copy()
    df_obj['area'] = (df_obj['width'] * df_obj['img_width']) * (df_obj['height'] * df_obj['img_height'])
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df_obj['area'], bins=50, kde=True)
    plt.title('Bounding Box Area Distribution (Full Dataset)', fontsize=16)
    plt.xlabel('Bounding Box Area (pixels^2)', fontsize=12)
    plt.xscale('log') # Log scale is essential for this plot
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

# --- 4. MAIN EXECUTION ---
def main():
    """Main function to run the complete, unified analysis with robust plotting."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"Analysis reports will be saved to '{REPORT_DIR}' directory.")
    
    # Step 1: Caching Logic
    if os.path.exists(CACHE_FILE):
        print(f"\n✅ Found cache file '{CACHE_FILE}'. Loading pre-parsed data...")
        combined_df = pd.read_pickle(CACHE_FILE)
        print("🎉 Data loaded instantly from cache.")
    else:
        print(f"\nℹ️ No cache file found. Starting full dataset parse (this will run only once)...")
        # DELETE THE CACHE IF IT MIGHT BE CORRUPTED
        # os.remove(CACHE_FILE) # Uncomment this line if you suspect a bad cache
        train_df = parse_yolo_data(TRAIN_DIR, CLASS_NAMES)
        val_df = parse_yolo_data(VAL_DIR, CLASS_NAMES)
        combined_df = pd.concat([train_df, val_df], ignore_index=True)
        print(f"\n💾 Saving parsed data to '{CACHE_FILE}' for next time...")
        combined_df.to_pickle(CACHE_FILE)
        print("✅ Data saved.")

    # Step 2: Diagnostic Check
    print("\n" + "="*50)
    print("      DIAGNOSTIC CHECK OF THE FINAL DATAFRAME")
    print("="*50)
    print("DataFrame Info:")
    combined_df.info()
    if combined_df.empty:
        print("\n❌ CRITICAL ERROR: The final DataFrame is empty. Cannot generate plots. Exiting.")
        print("Please check your dataset paths and ensure your label files are not all empty.")
        return
    print("\nDataFrame Head (first 5 rows):")
    print(combined_df.head())
    print("="*50)

    # Step 3: Generate and Save All Plots
    print("\n--- GENERATING UNIFIED VISUALIZATION REPORT ---")
    plot_class_distribution(combined_df, '1_class_distribution.png')
    plot_image_dimensions_separate(combined_df, '2_image_dimensions.png')
    plot_brightness_contrast_separate(combined_df, '3_brightness_contrast.png')
    plot_bbox_size_distribution(combined_df, '4_bbox_sizes.png')
    
    print("\n" + "="*50)
    print("✅ UNIFIED ANALYSIS COMPLETE!")
    print(f"All reports saved to '{REPORT_DIR}'.")
    print("="*50)

if __name__ == '__main__':
    main()