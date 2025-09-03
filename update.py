"""
DISCLAIMER - AMINDAV PROPERTY

This software is the exclusive property of Amindav. All rights reserved.
Unauthorized copying, distribution, or modification of this code is strictly prohibited.
This system is designed for fine-tuning AI models used in the bride and groom identification system.

SYSTEM OVERVIEW:
This simplified script performs model fine-tuning using pre-existing image data.
It loads a pre-trained CNN model, fine-tunes it on new data, and saves the updated weights.

WORKFLOW:
1. Loads pre-trained model weights
2. Trains the model on new image data with frozen early layers
3. Saves new model weights to specified path

AUTHOR: Amindav Development Team
VERSION: 1.0 (Simplified)
"""
import os
from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator


# --- USER CONFIGURATION - MODIFY THESE PATHS ---
current_model_path = Path("./models/crowd_classification.weights.h5")  # Set to None if training from scratch
updated_model_path = Path("./models/crowd_classification.weights.h5")  # Path for new weights
data_directory = Path("./data/training_data")  # Directory containing image subfolders

# --- Configuration Constants ---

# Model & Training Hyperparameters
# These parameters control the CNN architecture and training process
IMG_SIZE: Tuple[int, int] = (224, 224)  # Standard input size for the CNN model
BATCH_SIZE: int = 10  # Number of samples per training batch
NUM_CLASSES: int = 2  # Corresponds to the number of subfolders in DATA_DIR 
LEARNING_RATE: float = 1e-4  # Learning rate for fine-tuning (lower than initial training)
EPOCHS: int = 5  # Number of training epochs
VALIDATION_SPLIT: float = 0.01  # 1% of data for validation, as in original script


# --- Core Logic Functions (Unchanged) ---

def create_data_generators(dataset_dir: Path) -> Tuple[ImageDataGenerator, ImageDataGenerator]:
    """
    Creates and returns training and validation data generators.
    
    These generators handle data augmentation and preprocessing for the CNN model.
    They automatically split the data into training and validation sets.
    """
    datagen = ImageDataGenerator(
        rescale=1.0 / 255.0,  # Normalize pixel values to [0,1]
        validation_split=VALIDATION_SPLIT
    )

    train_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="training",
        shuffle=True,
    )

    val_generator = datagen.flow_from_directory(
        dataset_dir,
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        subset="validation",
        shuffle=True,  # As in original script
    )
    return train_generator, val_generator


def create_cnn_model(input_shape: Tuple, num_classes: int) -> models.Sequential:
    """
    Builds and returns the CNN model architecture.
    
    This creates a standard CNN architecture suitable for video frame classification.
    The model consists of convolutional layers, pooling layers, and dense layers.
    """
    model = models.Sequential([
        layers.Conv2D(32, (3, 3), activation="relu", input_shape=input_shape),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation="relu"),
        layers.Dense(num_classes, activation="softmax"),
    ])
    return model


def freeze_layers(model: models.Sequential, num_layers_to_freeze: int) -> models.Sequential:
    """
    Freezes the first `num_layers_to_freeze` layers of the model.
    
    This is a key technique in fine-tuning where early layers (which learn
    general features) are frozen while later layers are trained on new data.
    """
    for layer in model.layers[:num_layers_to_freeze]:
        layer.trainable = False
    print(f"INFO: First {num_layers_to_freeze} layers frozen for fine-tuning.")
    return model


def fine_tune_model(weights_path: Optional[Path], dataset_dir: Path) -> models.Sequential:
    """
    Orchestrates the model fine-tuning process, preserving the original logic.
    
    This function handles the complete fine-tuning workflow including model creation,
    weight loading, layer freezing, and training.
    """
    input_shape = (*IMG_SIZE, 3)
    
    # 1. Create model and load pre-trained weights if they exist
    model = create_cnn_model(input_shape, NUM_CLASSES)
    if weights_path and weights_path.exists():
        print(f"INFO: Loading pre-trained weights from: {weights_path}")
        model.load_weights(weights_path)
    else:
        print("INFO: No pre-trained weights specified or found. Training model from scratch.")

    # 2. Freeze earlier layers (EXACT LOGIC FROM ORIGINAL SCRIPT)
    if weights_path:
        num_layers_to_freeze = len(model.layers) - 6
        model = freeze_layers(model, num_layers_to_freeze)

    # 3. Recompile the model with a lower learning rate
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=LEARNING_RATE),
        loss="categorical_crossentropy",
        metrics=["accuracy"],
    )
    model.summary()

    # 4. Prepare data generators
    train_gen, val_gen = create_data_generators(dataset_dir)

    # 5. Fine-tune the model
    print("\nStarting model fine-tuning...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        verbose=1,
    )
    print("-> Fine-tuning complete.")
    return model


def main():
    """
    Main execution function to run the simplified workflow.
    
    Usage: Modify these paths before running:
    - current_model_path: Path to existing model weights (or None for training from scratch)
    - updated_model_path: Path where new model weights will be saved
    - data_directory: Path to directory containing image folders for training
    """
    print("--- Starting Simplified Fine-Tuning Script ---")
    

    
    # --- Validate paths ---
    if not data_directory.exists():
        print(f"ERROR: Data directory '{data_directory}' does not exist. Aborting.")
        return
    
    if not any(data_directory.iterdir()):
        print(f"ERROR: Data directory '{data_directory}' is empty. Aborting training.")
        return
    
    # Create output directory if it doesn't exist
    updated_model_path.parent.mkdir(parents=True, exist_ok=True)
    
    # --- Model Fine-Tuning ---
    print("\n--- Starting Model Fine-Tuning ---")
    try:
        fine_tuned_model = fine_tune_model(
            weights_path=current_model_path if current_model_path and current_model_path.exists() else None,
            dataset_dir=data_directory
        )
        
        # --- Save the Newly Trained Model ---
        fine_tuned_model.save_weights(updated_model_path)
        print(f"\n✅ Successfully saved fine-tuned model weights to: {updated_model_path}")
        
    except Exception as e:
        print(f"\nFATAL: An unexpected error occurred during model training: {e}")

    print("\n--- Script Finished ---")


if __name__ == "__main__":
    main()