# ==============================================================================
#                 YOLO Dataset Analysis Script 1: Labels-Only
#
# Description:
#   This script performs an extremely fast analysis by reading ONLY the .txt
#   label files. It calculates the most critical statistics like class
#   distribution and object counts without the slow process of loading images.
#
# Instructions:
#   1. Place this script in the same parent directory as your 'datasets' folder.
#   2. Ensure the CONFIGURATION variables below are correct.
#   3. Run from the terminal: python analyze_labels_only.py
#
# ==============================================================================

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# --- 1. CONFIGURATION ---
# Please ensure these settings match your project.
TRAIN_DIR = '../datasets/train'
VAL_DIR = '../datasets/val'
REPORT_DIR = '../report/label'
CLASS_NAMES = [
    'Motorcycle', 'Car', 'Bus', 'Truck', 'Transporter',
    'Container', 'Big_Transporter'
]

# --- 2. CORE PARSING FUNCTION ---
def parse_labels_only(data_dir, class_names):
    """Parses all .txt label files in a directory without reading images."""
    data = []
    if not os.path.isdir(data_dir):
        print(f"Error: Directory not found at '{data_dir}'.")
        return pd.DataFrame()

    label_files = [f for f in os.listdir(data_dir) if f.endswith('.txt')]
    print(f"Found {len(label_files)} label files in '{data_dir}'. Processing...")

    for label_file in label_files:
        image_name = os.path.splitext(label_file)[0] + '.jpg'
        with open(os.path.join(data_dir, label_file), 'r') as f:
            lines = f.readlines()
            # If the entire file is empty, record it as having no objects.
            if not lines:
                data.append([image_name, -1, 'no_object'])
            else:
                lines_processed = 0
                for line in lines:
                    # --- FIX IS HERE ---
                    # First, strip whitespace from the line.
                    stripped_line = line.strip()
                    # If the stripped line is not empty, then process it.
                    if stripped_line:
                        class_id = int(stripped_line.split()[0])
                        data.append([image_name, class_id, class_names[class_id]])
                        lines_processed += 1
                
                # If the file had lines but they were all blank, it also has no objects.
                if lines_processed == 0:
                    data.append([image_name, -1, 'no_object'])

    return pd.DataFrame(data, columns=['image_name', 'class_id', 'class_name'])

# --- 3. PLOTTING FUNCTIONS ---
def plot_class_distribution(df, title, filename):
    """Plots the total number of bounding boxes for each class."""
    print(f"  - Generating plot: {title}")
    plt.figure(figsize=(12, 7))
    class_counts = df[df['class_name'] != 'no_object']['class_name'].value_counts()
    sns.barplot(x=class_counts.index, y=class_counts.values, palette='viridis', order=class_counts.index)
    plt.title(title, fontsize=16)
    plt.xlabel('Class Name', fontsize=12)
    plt.ylabel('Number of Instances (Bounding Boxes)', fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_bboxes_per_image(df, title, filename):
    """Plots the distribution of how many objects appear per image."""
    print(f"  - Generating plot: {title}")
    bboxes_per_image = df[df['class_name'] != 'no_object'].groupby('image_name').size()
    plt.figure(figsize=(10, 6))
    sns.histplot(bboxes_per_image, bins=max(1, bboxes_per_image.max()), discrete=True)
    plt.title(title, fontsize=16)
    plt.xlabel('Number of Bounding Boxes per Image', fontsize=12)
    plt.ylabel('Frequency (Number of Images)', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

def plot_co_occurrence_matrix(df, class_names, title, filename):
    """Plots a heatmap of which classes appear together in the same image."""
    print(f"  - Generating plot: {title}")
    image_groups = df[df['class_name'] != 'no_object'].groupby('image_name')['class_name'].apply(list)
    co_matrix = pd.DataFrame(np.zeros((len(class_names), len(class_names))), index=class_names, columns=class_names)
    
    for classes_in_image in image_groups:
        unique_classes = sorted(list(set(classes_in_image)))
        for i in range(len(unique_classes)):
            for j in range(i + 1, len(unique_classes)):
                c1, c2 = unique_classes[i], unique_classes[j]
                co_matrix.loc[c1, c2] += 1
                co_matrix.loc[c2, c1] += 1

    plt.figure(figsize=(10, 8))
    sns.heatmap(co_matrix, annot=True, fmt=".0f", cmap="crest")
    plt.title(title, fontsize=16)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(os.path.join(REPORT_DIR, filename))
    plt.close()

# --- 4. MAIN EXECUTION ---
def main():
    """Main function to orchestrate the labels-only analysis."""
    os.makedirs(REPORT_DIR, exist_ok=True)
    print(f"Analysis reports will be saved to '{REPORT_DIR}'")
    
    # --- Parse Data ---
    print("\n--- Parsing Training Labels (Fast) ---")
    train_df = parse_labels_only(TRAIN_DIR, CLASS_NAMES)
    print("\n--- Parsing Validation Labels (Fast) ---")
    val_df = parse_labels_only(VAL_DIR, CLASS_NAMES)

    if train_df.empty or val_df.empty:
        print("\nCritical Error: Could not parse label files. Exiting.")
        return

    # --- Print Summary ---
    print("\n" + "="*50)
    print("      LABELS-ONLY DATASET OVERVIEW (EXACT COUNTS)")
    print("="*50)
    print(f"Total Images in Train Set: {train_df['image_name'].nunique()}")
    print(f"Total Bounding Boxes in Train Set: {len(train_df[train_df['class_name'] != 'no_object'])}")
    print(f"\nTotal Images in Validation Set: {val_df['image_name'].nunique()}")
    print(f"Total Bounding Boxes in Validation Set: {len(val_df[val_df['class_name'] != 'no_object'])}")
    print("="*50)

    # --- Generate Plots ---
    print("\n--- Generating Plots ---")
    plot_class_distribution(train_df, 'Class Distribution (Training Set)', '1_class_dist_train.png')
    plot_class_distribution(val_df, 'Class Distribution (Validation Set)', '1_class_dist_val.png')
    plot_bboxes_per_image(train_df, 'Objects per Image (Training Set)', '2_bboxes_per_image_train.png')
    plot_bboxes_per_image(val_df, 'Objects per Image (Validation Set)', '2_bboxes_per_image_val.png')
    plot_co_occurrence_matrix(train_df, CLASS_NAMES, 'Class Co-occurrence (Training Set)', '3_co_occurrence_train.png')

    print("\n✅ Labels-only analysis complete!")

if __name__ == '__main__':
    main()