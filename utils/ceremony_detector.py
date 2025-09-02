import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
from PIL import Image
from moviepy import VideoFileClip

# Define the model path here
MODEL_PATH = "models/cnn_model_weights13.weights.h5"


class VideoLabel:
    def __init__(self, img_size=(224, 224), num_classes=3, class_labels=None, n_frames=10):
        """
        Initialize the VideoLabel class.

        Args:
            img_size (tuple): Image size for resizing frames (default: (224, 224)).
            num_classes (int): Number of classes (default: 3).
            class_labels (dict): Dictionary mapping class indices to labels.
            n_frames (int): Number of frames to process (default: 10).
        """
        self.model_path = MODEL_PATH
        self.img_size = img_size
        self.num_classes = num_classes
        self.class_labels = class_labels or {0: "ceremony", 1: "dance", 2: "other"}
        self.n_frames = n_frames
        self.input_shape = (*img_size, 3)
        self.model = self._load_model()

    def _create_cnn_model(self):
        """Create the CNN model structure."""
        model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation="relu", input_shape=self.input_shape),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(128, (3, 3), activation="relu"),
            layers.MaxPooling2D((2, 2)),
            layers.Flatten(),
            layers.Dense(128, activation="relu"),
            layers.Dense(self.num_classes, activation="softmax"),
        ])
        model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
        return model

    def _load_model(self):
        """Load the model with weights."""
        model = self._create_cnn_model()
        model.load_weights(self.model_path)
        print(f"Model weights loaded successfully from '{self.model_path}'.")
        return model

    def _predict_frame(self, frame):
        """Predict the class of a single frame and return confidence scores."""
        frame_array = frame / 255.0
        frame_array = np.expand_dims(frame_array, axis=0)

        pred = self.model.predict(frame_array, verbose=0)[0]
        predicted_class = np.argmax(pred)
        confidence_scores = {self.class_labels[i]: prob for i, prob in enumerate(pred)}
        predicted_label = self.class_labels.get(predicted_class, "other")

        return predicted_label, confidence_scores

    def label_video(self, video_path):
        """Process the video and predict labels for frames."""
        clip = VideoFileClip(video_path)
        timestamps = np.linspace(0, clip.duration, self.n_frames, endpoint=False)

        dance, ceremony, other = 0, 0, 0
        for t in timestamps:
            frame = clip.get_frame(t)
            frame = Image.fromarray(frame).convert("RGB").resize(self.img_size)
            label, confidences = self._predict_frame(np.array(frame))
            print(f"Timestamp {t:.2f}s: Label = {label}, Confidence Scores = {confidences}")

            if label == "dance":
                dance += 1
            elif label == "ceremony":
                ceremony += 1
            else:
                other += 1

        clip.close()
        return dance, ceremony, other

    def process_video_list(self, video_paths):
        """
        Process a list of videos and return those classified as ceremony.
        
        Args:
            video_paths (list): List of video file paths to process
            
        Returns:
            list: List of video paths that are classified as ceremony
        """
        ceremony_videos = []
        
        print(f"Processing {len(video_paths)} videos for ceremony classification...")
        
        for i, video_path in enumerate(video_paths, 1):
            print(f"\nProcessing video {i}/{len(video_paths)}: {video_path}")
            
            try:
                dance_count, ceremony_count, other_count = self.label_video(video_path)
                
                # Calculate percentages
                total_frames = dance_count + ceremony_count + other_count
                ceremony_percentage = (ceremony_count / total_frames) * 100 if total_frames > 0 else 0
                
                print(f"Results for {video_path}:")
                print(f"  - Dance frames: {dance_count}")
                print(f"  - Ceremony frames: {ceremony_count}")
                print(f"  - Other frames: {other_count}")
                print(f"  - Ceremony percentage: {ceremony_percentage:.1f}%")
                
                # Classify as ceremony if more than 50% of frames are ceremony
                if ceremony_percentage > 50:
                    ceremony_videos.append(video_path)
                    print(f"  ✓ Classified as CEREMONY")
                else:
                    print(f"  ✗ Not classified as ceremony")
                    
            except Exception as e:
                print(f"Error processing video {video_path}: {e}")
                continue
        
        print(f"\nFound {len(ceremony_videos)} ceremony videos out of {len(video_paths)} processed")
        return ceremony_videos
