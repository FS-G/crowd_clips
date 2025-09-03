import os
import numpy as np
from pathlib import Path
from moviepy import VideoFileClip
from tensorflow.keras import layers, models
from PIL import Image


MODEL_PATH = "models/crowd_classification.weights.h5"

class CrowdClipExporter:
    def __init__(self, output_path, resolution="1920x1080", duration_seconds=5, skip_frame_rate=5):
        """
        Initialize the CrowdClipExporter class.
        
        Args:
            output_path (str): Directory path where clips will be exported
            resolution (str): Output resolution in format "widthxheight" (default: "1920x1080")
            duration_seconds (int): Duration of each clip in seconds (default: 5)
            skip_frame_rate (int): Number of frames to skip between analysis (default: 5)
        """
        self.output_path = output_path
        self.resolution = resolution
        self.duration_seconds = duration_seconds
        self.skip_frame_rate = skip_frame_rate
        
        # Crowd detection model settings
        self.crowd_model_path = MODEL_PATH
        self.img_size = (224, 224)
        self.num_classes = 2
        self.class_labels = {0: "crowd", 1: "non_crowd"}
        self.input_shape = (*self.img_size, 3)
        self.consecutive_crowd_frames = 5  # Number of consecutive crowd frames needed
        
        # Load crowd detection model
        self.crowd_model = self._load_crowd_model()
        
        # Parse resolution
        try:
            width, height = map(int, resolution.split('x'))
            self.width = width
            self.height = height
        except ValueError:
            raise ValueError(f"Invalid resolution format: {resolution}. Expected format: 'widthxheight'")
        
        # Create output directory if it doesn't exist
        os.makedirs(self.output_path, exist_ok=True)
        
        # Test if directory is writable
        test_file = os.path.join(self.output_path, "test_write.txt")
        try:
            with open(test_file, 'w') as f:
                f.write("test")
            os.remove(test_file)
            print(f"✓ Output directory is writable: {self.output_path}")
        except Exception as e:
            print(f"✗ Warning: Output directory may not be writable: {e}")
        
        print(f"Resolution: {self.resolution}")
        print(f"Clip duration: {self.duration_seconds} seconds")
        print(f"Skip frame rate: {self.skip_frame_rate}")
        print(f"Consecutive crowd frames needed: {self.consecutive_crowd_frames}")
    
    def _create_crowd_cnn_model(self):
        """Create the CNN model structure for crowd detection."""
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
    
    def _load_crowd_model(self):
        """Load the crowd detection model with weights."""
        try:
            model = self._create_crowd_cnn_model()
            model.load_weights(self.crowd_model_path)
            print(f"✓ Crowd detection model loaded successfully from '{self.crowd_model_path}'")
            return model
        except Exception as e:
            print(f"✗ Error loading crowd detection model: {e}")
            print("⚠ Crowd detection will be disabled")
            return None
    
    def _predict_crowd_frame(self, frame):
        """Predict if a single frame contains crowd."""
        if self.crowd_model is None:
            return "non_crowd", {"non_crowd": 0.5, "crowd": 0.5}
        
        frame_array = frame / 255.0
        frame_array = np.expand_dims(frame_array, axis=0)
        
        pred = self.crowd_model.predict(frame_array, verbose=0)[0]
        predicted_class = np.argmax(pred)
        confidence_scores = {self.class_labels[i]: prob for i, prob in enumerate(pred)}
        predicted_label = self.class_labels.get(predicted_class, "non_crowd")
        
        return predicted_label, confidence_scores
    

    
    def _generate_output_filename(self, video_path, start_time, segment_index=None):
        """
        Generate a unique output filename for the clip.
        
        Args:
            video_path (str): Original video file path
            start_time (float): Start time of the clip
            segment_index (int, optional): Index of the crowd segment
            
        Returns:
            str: Output filename
        """
        # Get original filename without extension
        video_name = Path(video_path).stem
        
        # Create filename with start time and segment index
        if segment_index:
            output_filename = f"{video_name}_clip_{start_time:.1f}s_seg{segment_index}.mp4"
        else:
            output_filename = f"{video_name}_clip_{start_time:.1f}s.mp4"
        
        # Ensure filename is valid
        output_filename = "".join(c for c in output_filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
        output_filename = output_filename.replace(' ', '_')
        
        return output_filename
    
    def _find_crowd_segments(self, video_path):
        """
        Find segments in the video that contain consecutive crowd frames.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            list: List of tuples (start_time, end_time) for crowd segments
        """
        if self.crowd_model is None:
            print("  ⚠ Crowd detection model not available, using random clip")
            return []
        
        try:
            print(f"  Analyzing video for crowd segments...")
            
            with VideoFileClip(video_path) as clip:
                video_duration = clip.duration
                fps = clip.fps
                
                # Calculate frame timestamps based on skip_frame_rate
                frame_interval = self.skip_frame_rate / fps
                timestamps = np.arange(0, video_duration, frame_interval)
                
                print(f"    Video duration: {video_duration:.2f}s, FPS: {fps:.2f}")
                print(f"    Analyzing {len(timestamps)} frames (every {self.skip_frame_rate} frames)")
                
                crowd_segments = []
                consecutive_crowd_count = 0
                crowd_start_time = None
                
                for i, timestamp in enumerate(timestamps):
                    try:
                        # Extract frame
                        frame = clip.get_frame(timestamp)
                        frame = Image.fromarray(frame).convert("RGB").resize(self.img_size)
                        
                        # Predict crowd
                        label, confidences = self._predict_crowd_frame(np.array(frame))
                        
                        if i % 10 == 0:  # Print every 10th frame to avoid spam
                            print(f"    Frame {i+1}/{len(timestamps)} at {timestamp:.2f}s: {label} ({confidences['crowd']:.2f})")
                        
                        # Check for consecutive crowd frames
                        if label == "crowd":
                            if crowd_start_time is None:
                                crowd_start_time = timestamp
                            consecutive_crowd_count += 1
                        else:
                            # Reset if we have enough consecutive crowd frames
                            if consecutive_crowd_count >= self.consecutive_crowd_frames:
                                end_time = min(timestamp, video_duration)
                                crowd_segments.append((crowd_start_time, end_time))
                                print(f"    ✓ Found crowd segment: {crowd_start_time:.2f}s - {end_time:.2f}s")
                            
                            # Reset counters
                            consecutive_crowd_count = 0
                            crowd_start_time = None
                    
                    except Exception as e:
                        print(f"    ⚠ Error processing frame at {timestamp:.2f}s: {e}")
                        continue
                
                # Check if we have a crowd segment at the end
                if consecutive_crowd_count >= self.consecutive_crowd_frames:
                    end_time = video_duration
                    crowd_segments.append((crowd_start_time, end_time))
                    print(f"    ✓ Found crowd segment: {crowd_start_time:.2f}s - {end_time:.2f}s")
                
                print(f"    Found {len(crowd_segments)} crowd segments")
                return crowd_segments
                
        except Exception as e:
            print(f"  ✗ Error analyzing video for crowd segments: {e}")
            return []
    
    def export_clip_from_video(self, video_path):
        """
        Export clips from a video based on crowd detection segments.
        Only exports if crowd segments exist and are long enough.
        
        Args:
            video_path (str): Path to the input video file
            
        Returns:
            list: List of exported clip paths, or empty list if no valid segments
        """
        try:
            print(f"Processing video: {video_path}")
            
            # Load video
            with VideoFileClip(video_path) as clip:
                video_duration = clip.duration
                print(f"  Video duration: {video_duration:.2f} seconds")
                
                # Check if video is long enough
                if video_duration < self.duration_seconds:
                    print(f"  Warning: Video too short ({video_duration:.2f}s < {self.duration_seconds}s), skipping...")
                    return []
                
                # Find crowd segments
                crowd_segments = self._find_crowd_segments(video_path)
                
                if not crowd_segments:
                    print(f"  ⚠ No crowd segments found, skipping video")
                    return []
                
                exported_clips = []
                print(f"  Found {len(crowd_segments)} crowd segments")
                
                # Process each crowd segment
                for i, (start_time, end_time) in enumerate(crowd_segments):
                    segment_duration = end_time - start_time
                    
                    # Skip segments that are too short
                    if segment_duration < self.duration_seconds:
                        print(f"    Segment {i+1}: Too short ({segment_duration:.2f}s < {self.duration_seconds}s), skipping")
                        continue
                    
                    # Take exactly duration_seconds from the start of the segment
                    clip_start = start_time
                    clip_end = start_time + self.duration_seconds
                    
                    print(f"    Segment {i+1}: Using {clip_start:.2f}s - {clip_end:.2f}s (exactly {self.duration_seconds}s)")
                    
                    # Extract subclip
                    subclip = clip.subclipped(clip_start, clip_end)
                    
                    # Resize to target resolution
                    subclip = subclip.resized((self.width, self.height))
                    
                    # Generate output filename with segment index
                    output_filename = self._generate_output_filename(video_path, clip_start, i+1)
                    output_path = os.path.join(self.output_path, output_filename)
                    
                    # Export the clip
                    print(f"      Exporting to: {output_path}")
                    
                    # Ensure output directory exists
                    os.makedirs(os.path.dirname(output_path), exist_ok=True)
                    
                    # Write the video file with explicit parameters
                    subclip.write_videofile(
                        output_path,
                        codec='libx264',
                        audio_codec='aac',
                        temp_audiofile='temp-audio.m4a',
                        remove_temp=True,
                        fps=24  # Set explicit FPS
                    )
                    
                    # Close the subclip to free memory
                    subclip.close()
                    
                    # Wait a moment for file system to update
                    import time
                    time.sleep(0.5)
                    
                    # Verify the file was actually saved
                    if os.path.exists(output_path):
                        file_size = os.path.getsize(output_path)
                        if file_size > 0:
                            print(f"      ✓ Successfully exported: {output_filename} ({file_size:,} bytes)")
                            exported_clips.append(output_path)
                        else:
                            print(f"      ✗ Error: File was created but is empty: {output_path}")
                            # Try to remove empty file
                            try:
                                os.remove(output_path)
                            except:
                                pass
                    else:
                        print(f"      ✗ Error: File was not saved to {output_path}")
                
                print(f"  ✓ Exported {len(exported_clips)} clips from {len(crowd_segments)} segments")
                return exported_clips
                
        except Exception as e:
            print(f"  ✗ Error processing video {video_path}: {e}")
            import traceback
            print(f"  Error details: {traceback.format_exc()}")
            return []
    
    def export_clips_from_videos(self, video_paths):
        """
        Export clips from a list of videos.
        
        Args:
            video_paths (list): List of video file paths to process
            
        Returns:
            list: List of successfully exported clip paths
        """
        exported_clips = []
        
        print(f"\nStarting clip export for {len(video_paths)} videos...")
        print(f"Output directory: {self.output_path}")
        
        for i, video_path in enumerate(video_paths, 1):
            print(f"\n[{i}/{len(video_paths)}] Processing: {os.path.basename(video_path)}")
            
            exported_paths = self.export_clip_from_video(video_path)
            if exported_paths:
                exported_clips.extend(exported_paths)
        
        print(f"\n✓ Export complete! Successfully exported {len(exported_clips)} clips out of {len(video_paths)} videos")
        
        if exported_clips:
            total_size = 0
            print("\nExported clips:")
            for clip_path in exported_clips:
                file_size = os.path.getsize(clip_path)
                total_size += file_size
                print(f"  - {os.path.basename(clip_path)} ({file_size:,} bytes)")
            
            print(f"\nTotal exported size: {total_size:,} bytes ({total_size / (1024*1024):.1f} MB)")
        
        return exported_clips
    
    def test_video_processing(self, test_video_path):
        """
        Test method to verify video processing works with a single video.
        
        Args:
            test_video_path (str): Path to a test video file
            
        Returns:
            bool: True if processing works, False otherwise
        """
        print(f"\nTesting video processing with: {test_video_path}")
        
        if not os.path.exists(test_video_path):
            print(f"✗ Test video does not exist: {test_video_path}")
            return False
        
        try:
            # Test video loading
            with VideoFileClip(test_video_path) as clip:
                print(f"✓ Video loaded successfully. Duration: {clip.duration:.2f}s")
                print(f"✓ Video size: {clip.size}")
                print(f"✓ Video fps: {clip.fps}")
            
            # Test clip export
            results = self.export_clip_from_video(test_video_path)
            if results:
                print(f"✓ Test successful! {len(results)} clip(s) saved:")
                for result in results:
                    print(f"  - {result}")
                return True
            else:
                print(f"✗ Test failed! No clips were saved.")
                return False
                
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            print(f"Error details: {traceback.format_exc()}")
            return False
