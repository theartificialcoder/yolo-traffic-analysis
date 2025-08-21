import os
import shutil
import torch
from ultralytics import YOLO

# =================================================================================================
# ---  USER CONFIGURATION  ---
# =================================================================================================

# 1. BASE MODEL FOR TRANSFER LEARNING
#    Choose the pre-trained model to start from. You selected 'yolov8l.pt' for high accuracy.
#    Options: 'yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt'
BASE_MODEL = 'yolov8m.pt'

# 2. DATASET CONFIGURATION FILE
#    Path to your dataset.yaml file. This file tells YOLO where your data is and what the classes are.
DATASET_CONFIG = '../datasets/dataset.yaml'

# 3. TRAINING HYPERPARAMETERS
#    - epochs: Number of times to loop through the entire dataset. 100 is a good starting point.
#    - imgsz: The image size the model will be trained on. 640 is standard for YOLOv8.
#             Larger sizes (e.g., 1280) can improve accuracy for small objects but require more VRAM.
EPOCHS = 100
IMAGE_SIZE = 640
BATCH_SIZE = 24 # Adjust based on your GPU memory. 16 is a good starting point for most GPUs.
# 4. OUTPUT CONFIGURATION
#    - project: The name of the main output directory for all experimental runs.
#    - experiment_name: A unique, descriptive name for this specific training run. This will also be
#                       used for your final, cleaned model file name.
#    - final_model_dir: The clean, top-level directory where your best model will be saved.
PROJECT_NAME = 'testrun'
EXPERIMENT_NAME = 'yolov8s_traffic_default' # Descriptive name for the run
FINAL_MODEL_DIR = '../models'

# =================================================================================================
# ---  SCRIPT LOGIC  ---
# =================================================================================================

def train_model():
    """
    Main function to run the YOLOv8 training process.
    """
    print("--- Starting YOLOv8 Training ---")
    
    # --- Device Check ---
    # Check for GPU and print what device is being used.
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Using device: {device}")
    if device == 'cpu':
        print("WARNING: Training on CPU is very slow. A CUDA-enabled GPU is highly recommended.")

    # --- Initialize Model ---
    # Load the specified pre-trained model from Ultralytics.
    print(f"Loading base model: {BASE_MODEL}")
    model = YOLO(BASE_MODEL)

    # --- Start Training ---
    # The 'model.train()' function starts the training process.
    # It returns a 'results' object containing information about the run.
    print(f"Starting training for {EPOCHS} epochs with image size {IMAGE_SIZE}...")
    results = model.train(
        data=DATASET_CONFIG,
        epochs=EPOCHS,
        imgsz=IMAGE_SIZE,
        batch = BATCH_SIZE,
        project=PROJECT_NAME,
        name=EXPERIMENT_NAME,
        exist_ok=True  # Allows overwriting a previous run with the same name
    )
    print("--- Training Complete ---")

    # --- Save the Best Model Cleanly ---
    # The best performing model is saved as 'best.pt' in the experiment's output directory.
    # We will copy it to our clean 'models/' directory for easy access.
    
    print("--- Saving Final Model ---")
    # Ensure the final, clean models directory exists.
    os.makedirs(FINAL_MODEL_DIR, exist_ok=True)

    # Path to the best model weights from the training run. 'results.save_dir' contains the
    # path to the experiment folder (e.g., 'training_runs/yolov8l_traffic_v1_100e').
    best_model_path = os.path.join(results.save_dir, 'weights/best.pt')

    # Define the new, descriptive path for the final model.
    final_model_name = f"{EXPERIMENT_NAME}.pt"
    final_model_path = os.path.join(FINAL_MODEL_DIR, final_model_name)

    # Check if the best model was created before trying to copy it.
    if os.path.exists(best_model_path):
        shutil.copyfile(best_model_path, final_model_path)
        print(f"Successfully copied best model to: {final_model_path}")
    else:
        print(f"ERROR: Could not find best model at '{best_model_path}'. Check training results.")
        
    print("--- Process Finished ---")


# This ensures the script runs only when executed directly (not when imported)
if __name__ == '__main__':
    train_model()