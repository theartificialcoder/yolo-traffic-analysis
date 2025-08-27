import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from collections import Counter

# ==============================================================================
# --- CONFIGURATION ---
# This section has been updated based on your dataset.yaml and structure.
# ==============================================================================
# Paths to your dataset folders (relative to the script's location)
# Assumes 'train' and 'val' folders are inside a 'datasets' directory.
TRAIN_DIR = '../datasets/train'
VAL_DIR = '../datasets/val'

# Class names extracted from your dataset.yaml
CLASS_NAMES = [
    'Motorcycle', 'Car', 'Bus', 'Truck', 'Transporter',
    'Container', 'Big_Transporter'
]

# Directory to save the output charts
REPORT_DIR = '../report'
# ==============================================================================

# Ensure the report directory exists
os.makedirs(REPORT_DIR, exist_ok=True)

def parse_yolo_data_mixed_folder(data_dir, class_names):
    """
    Parses all image and label files in a given directory.

    This function iterates through every image, reads its properties (like size and
    brightness), and then reads the corresponding YOLO label file to extract
    bounding box information. It includes a progress indicator to provide feedback
    during this long process.

    Args:
        data_dir (str): The path to the directory containing both .jpg and .txt files.
        class_names (list): A list of strings for the class names.

    Returns:
        pandas.DataFrame: A DataFrame containing all the parsed data, with each
                          row representing a single bounding box.
    """
    data = []

    # --- Pre-computation and Checks ---
    # First, check if the directory actually exists to prevent errors.
    if not os.path.isdir(data_dir):
        print(f"❌ Error: Directory not found at '{data_dir}'. Please check the configuration paths.")
        return pd.DataFrame() # Return an empty DataFrame to avoid crashing

    # Get a list of all image files to process.
    image_files = [f for f in os.listdir(data_dir) if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
    total_images = len(image_files)

    # Check if any images were found.
    if not image_files:
        print(f"⚠️ Warning: No images found in '{data_dir}'.")
        return pd.DataFrame()

    print(f"✅ Found {total_images} images in '{data_dir}'. Starting processing...")

    # --- Main Processing Loop ---
    # Iterate over each image with an index for progress tracking.
    for i, img_file in enumerate(image_files):

        # --- PROGRESS INDICATOR ---
        # This print statement provides real-time feedback on the script's progress.
        # It prints on the first image, the last image, and every 20th image in between.
        if (i + 1) % 20 == 0 or i == 0 or i == total_images - 1:
            print(f"  ➡️ Processing image [{i + 1}/{total_images}]: {img_file}")

        # Construct the full paths for the image and its corresponding label file.
        img_path = os.path.join(data_dir, img_file)
        label_path = os.path.join(data_dir, os.path.splitext(img_file)[0] + '.txt')

        # --- Image Analysis ---
        # Read the image using OpenCV.
        image = cv2.imread(img_path)
        if image is None:
            print(f"  - ⚠️ Warning: Could not read image '{img_path}'. Skipping.")
            continue # Skip to the next image if this one is corrupt or unreadable

        # Get image dimensions (height, width).
        height, width, _ = image.shape
        # Convert to grayscale for brightness/contrast calculation. This is faster and more standard.
        gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # Brightness is the average pixel intensity.
        brightness = np.mean(gray_image)
        # Contrast is the standard deviation of pixel intensities.
        contrast = np.std(gray_image)

        # --- Label File Parsing ---
        # Check if a corresponding label file exists.
        if os.path.exists(label_path):
            with open(label_path, 'r') as f:
                lines = f.readlines()
                # Handle cases where the label file is empty.
                if not lines:
                    # Append a row indicating an image with no objects.
                    data.append([img_path, width, height, brightness, contrast, -1, 'no_object', 0, 0, 0, 0])
                else:
                    # Read each line, which corresponds to one bounding box.
                    for line in lines:
                        parts = line.strip().split()
                        if len(parts) < 5:
                            print(f"  - ⚠️ Warning: Malformed line in '{label_path}'. Skipping line.")
                            continue # Skip corrupted or incomplete lines
                        
                        # YOLO format: class_id, x_center, y_center, width, height
                        class_id = int(parts[0])
                        x_center, y_center, w, h = map(float, parts[1:])
                        
                        # Append one row per bounding box to our data list.
                        data.append([img_path, width, height, brightness, contrast, class_id, class_names[class_id], x_center, y_center, w, h])
        else:
            # If no label file is found for an image, it's considered an image with no objects.
            data.append([img_path, width, height, brightness, contrast, -1, 'no_object', 0, 0, 0, 0])

    # --- Final DataFrame Creation ---
    # Define the column names for our dataset.
    columns = ['image_path', 'img_width', 'img_height', 'brightness', 'contrast', 'class_id', 'class_name', 'x_center', 'y_center', 'width', 'height']
    # Create the final pandas DataFrame.
    print(f"🎉 Finished parsing {data_dir}.")
    return pd.DataFrame(data, columns=columns)

# ==============================================================================
# --- 3. PLOTTING FUNCTIONS ---
# Each function is responsible for creating and saving one type of chart.
# ==============================================================================

def plot_class_distribution(df, title, filename):
    """Generates a bar chart of instance counts for each class."""
    print(f"  - Generating plot: {title}")
    plt.figure(figsize=(12, 7))
    # Filter out any entries corresponding to images with no objects.
    class_counts = df[df['class_name'] != 'no_object']['class_name'].value_counts()
    # Create the bar plot using seaborn.
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis', order=class_counts.index)
    # Formatting
    plt.title(title, fontsize=16)
    plt.xlabel('Class Name', fontsize=12)
    plt.ylabel('Number of Instances (Bounding Boxes)', fontsize=12)
    plt.xticks(rotation=45, ha='right') # Rotate labels for better readability
    plt.tight_layout()
    # Save the figure to the report directory.
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_images_per_class(df, title, filename):
    """Generates a bar chart of unique images containing each class."""
    print(f"  - Generating plot: {title}")
    plt.figure(figsize=(12, 7))
    # Group by class name and count the number of unique image paths.
    images_per_class = df[df['class_name'] != 'no_object'].groupby('class_name')['image_path'].nunique().sort_values(ascending=False)
    sns.barplot(x=images_per_class.index, y=images_per_class.values, palette='plasma', order=images_per_class.index)
    plt.title(title, fontsize=16)
    plt.xlabel('Class Name', fontsize=12)
    plt.ylabel('Number of Unique Images', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_co_occurrence_matrix(df, class_names, title, filename):
    """Generates a heatmap showing which classes appear together in the same image."""
    print(f"  - Generating plot: {title}")
    df_filtered = df[df['class_name'] != 'no_object']
    # Get a list of classes for each image.
    image_groups = df_filtered.groupby('image_path')['class_name'].apply(list).reset_index()
    # Filter out images that have only one object, as co-occurrence is not possible.
    image_groups = image_groups[image_groups['class_name'].apply(len) > 1]
    
    # Initialize an empty matrix with class names as index and columns.
    num_classes = len(class_names)
    co_occurrence_matrix = pd.DataFrame(np.zeros((num_classes, num_classes)), index=class_names, columns=class_names)
    
    # Iterate through each image's class list to populate the matrix.
    for _, row in image_groups.iterrows():
        classes_in_image = sorted(list(set(row['class_name'])))
        for i in range(len(classes_in_image)):
            for j in range(i, len(classes_in_image)):
                class1, class2 = classes_in_image[i], classes_in_image[j]
                if class1 in co_occurrence_matrix.index and class2 in co_occurrence_matrix.columns and class1 != class2:
                    # Increment count for both pairs (e.g., car-truck and truck-car).
                    co_occurrence_matrix.loc[class1, class2] += 1
                    co_occurrence_matrix.loc[class2, class1] += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(co_occurrence_matrix, annot=True, fmt=".0f", cmap="crest")
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=0)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_bbox_size_distribution(df, title, filename):
    """Generates a histogram of bounding box areas to show object size distribution."""
    print(f"  - Generating plot: {title}")
    df_obj = df[df['class_name'] != 'no_object'].copy()
    if df_obj.empty:
        print(f"  - ⚠️ Skipping '{filename}': No objects to plot.")
        return
    # YOLO dimensions are relative; convert them to absolute pixel values.
    df_obj['abs_width'] = df_obj['width'] * df_obj['img_width']
    df_obj['abs_height'] = df_obj['height'] * df_obj['img_height']
    df_obj['area'] = df_obj['abs_width'] * df_obj['abs_height']
    
    plt.figure(figsize=(10, 6))
    sns.histplot(df_obj['area'], bins=50, kde=True)
    plt.title(title, fontsize=16)
    plt.xlabel('Bounding Box Area (pixels^2)', fontsize=12)
    plt.ylabel('Frequency', fontsize=12)
    # A log scale is crucial here, as it prevents large objects from squashing
    # the visualization of small objects, making the whole distribution visible.
    plt.xscale('log')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_bbox_aspect_ratio(df, title, filename):
    """Generates a scatter plot of bounding box widths vs. heights."""
    print(f"  - Generating plot: {title}")
    df_obj = df[df['class_name'] != 'no_object'].copy()
    if df_obj.empty:
        print(f"  - ⚠️ Skipping '{filename}': No objects to plot.")
        return
    df_obj['abs_width'] = df_obj['width'] * df_obj['img_width']
    df_obj['abs_height'] = df_obj['height'] * df_obj['img_height']

    plt.figure(figsize=(8, 8))
    # For very large datasets, plotting every point can be slow and messy.
    # We take a random sample of up to 5000 points for a clean visualization.
    sample_df = df_obj.sample(min(len(df_obj), 5000))
    sns.scatterplot(x='abs_width', y='abs_height', data=sample_df, alpha=0.5)
    plt.title(title, fontsize=16)
    plt.xlabel('Bounding Box Width (pixels)', fontsize=12)
    plt.ylabel('Bounding Box Height (pixels)', fontsize=12)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_bboxes_per_image(df, title, filename):
    """Generates a histogram showing the number of objects per image."""
    print(f"  - Generating plot: {title}")
    df_obj = df[df['class_name'] != 'no_object']
    if df_obj.empty:
        print(f"  - ⚠️ Skipping '{filename}': No objects to plot.")
        return
    # Group by image and count the number of bounding boxes.
    bboxes_per_image = df_obj.groupby('image_path').size()
    
    plt.figure(figsize=(10, 6))
    # Using discrete bins makes this a proper histogram for integer counts.
    sns.histplot(bboxes_per_image, bins=max(1, bboxes_per_image.max()), discrete=True)
    plt.title(title, fontsize=16)
    plt.xlabel('Number of Bounding Boxes per Image', fontsize=12)
    plt.ylabel('Frequency (Number of Images)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_image_dimensions(df, title, filename):
    """Generates histograms for the width and height of images."""
    print(f"  - Generating plot: {title}")
    # Get unique image dimensions to avoid recounting the same image.
    dims = df[['image_path', 'img_width', 'img_height']].drop_duplicates()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)
    
    sns.histplot(dims['img_width'], ax=axes[0], bins=30, kde=True, color='skyblue')
    axes[0].set_title('Image Width Distribution')
    axes[0].set_xlabel('Width (pixels)')
    
    sns.histplot(dims['img_height'], ax=axes[1], bins=30, kde=True, color='salmon')
    axes[1].set_title('Image Height Distribution')
    axes[1].set_xlabel('Height (pixels)')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_brightness_contrast(df, title, filename):
    """Generates histograms for the brightness and contrast of images."""
    print(f"  - Generating plot: {title}")
    dims = df[['image_path', 'brightness', 'contrast']].drop_duplicates()
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle(title, fontsize=16)
    
    sns.histplot(dims['brightness'], ax=axes[0], bins=30, kde=True, color='orange')
    axes[0].set_title('Brightness Distribution')
    axes[0].set_xlabel('Average Pixel Intensity (0-255)')
    
    sns.histplot(dims['contrast'], ax=axes[1], bins=30, kde=True, color='purple')
    axes[1].set_title('Contrast Distribution (Std Dev)')
    axes[1].set_xlabel('Standard Deviation of Pixel Intensity')
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

# ==============================================================================
# --- 4. MAIN EXECUTION SCRIPT ---
# This is the main function that orchestrates the entire analysis process.
# ==============================================================================

def main():
    """Main function to run the complete analysis from start to finish."""
    
    # Create the directory for saving reports if it doesn't already exist.
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"Analysis reports will be saved to the '{REPORT_DIR}' directory.")

    # --- Step 1: Load and Parse Data ---
    print("\n" + "="*50)
    print("STEP 1: PARSING DATASET FILES")
    print("="*50)
    train_df = parse_yolo_data_mixed_folder(TRAIN_DIR, CLASS_NAMES)
    val_df = parse_yolo_data_mixed_folder(VAL_DIR, CLASS_NAMES)

    # --- Early Exit ---
    # If parsing failed (e.g., wrong path), stop the script gracefully.
    if train_df.empty or val_df.empty:
        print("\n❌ Critical Error: Could not parse one or both datasets. Please check paths and file structures. Exiting.")
        return

    # --- Step 2: Display High-Level Overview ---
    print("\n" + "="*50)
    print("STEP 2: DATASET OVERVIEW STATISTICS")
    print("="*50)
    print(f"Total Classes: {len(CLASS_NAMES)} -> {CLASS_NAMES}")
    print("-" * 50)
    print("--- Training Set ---")
    print(f"Total Unique Images: {train_df['image_path'].nunique()}")
    print(f"Total Bounding Boxes: {len(train_df[train_df['class_name'] != 'no_object'])}")
    print("\n--- Validation Set ---")
    print(f"Total Unique Images: {val_df['image_path'].nunique()}")
    print(f"Total Bounding Boxes: {len(val_df[val_df['class_name'] != 'no_object'])}")
    print("="*50)

    # --- Step 3: Generate and Save All Plots ---
    print("\n" + "="*50)
    print("STEP 3: GENERATING VISUALIZATION REPORT")
    print("="*50)

    # Generate plots for both training and validation sets for comparison.
    for df, set_name in [(train_df, 'Training'), (val_df, 'Validation')]:
        print(f"\n--- Generating plots for {set_name} Set ---")
        plot_class_distribution(df, f'Class Distribution ({set_name} Set)', f'1_class_dist_{set_name.lower()}.png')
        plot_images_per_class(df, f'Unique Images per Class ({set_name} Set)', f'2_images_per_class_{set_name.lower()}.png')
        plot_bbox_size_distribution(df, f'Bounding Box Area Distribution ({set_name} Set)', f'4_bbox_area_{set_name.lower()}.png')
        plot_bbox_aspect_ratio(df, f'Bounding Box Aspect Ratios ({set_name} Set)', f'5_bbox_aspect_ratio_{set_name.lower()}.png')
        plot_bboxes_per_image(df, f'Number of Objects per Image ({set_name} Set)', f'6_bboxes_per_image_{set_name.lower()}.png')
        plot_image_dimensions(df, f'Image Dimensions ({set_name} Set)', f'7_image_dims_{set_name.lower()}.png')
        plot_brightness_contrast(df, f'Image Brightness & Contrast ({set_name} Set)', f'8_brightness_contrast_{set_name.lower()}.png')

    # Co-occurrence is usually most interesting for the training set.
    print("\n--- Generating Co-occurrence Matrix for Training Set ---")
    plot_co_occurrence_matrix(train_df, CLASS_NAMES, 'Class Co-occurrence Matrix (Training Set)', '3_co_occurrence_train.png')

    print("\n" + "="*50)
    print("✅ ANALYSIS COMPLETE!")
    print(f"All reports have been saved to the '{REPORT_DIR}' directory.")
    print("="*50)

# This standard Python construct ensures that the main() function is called
# only when the script is executed directly.
if __name__ == '__main__':
    main()