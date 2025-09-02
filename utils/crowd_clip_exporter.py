import os
import random
from pathlib import Path
from moviepy import VideoFileClip


class CrowdClipExporter:
    def __init__(self, output_path, resolution="1920x1080", duration_seconds=5):
        """
        Initialize the CrowdClipExporter class.
        
        Args:
            output_path (str): Directory path where clips will be exported
            resolution (str): Output resolution in format "widthxheight" (default: "1920x1080")
            duration_seconds (int): Duration of each clip in seconds (default: 5)
        """
        self.output_path = output_path
        self.resolution = resolution
        self.duration_seconds = duration_seconds
        
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
    
    def _get_random_start_time(self, video_duration):
        """
        Get a random start time that ensures the clip doesn't start at the beginning or end.
        
        Args:
            video_duration (float): Total duration of the video in seconds
            
        Returns:
            float: Random start time in seconds
        """
        # Ensure we have enough time for the clip
        if video_duration <= self.duration_seconds:
            return 0
        
        # Calculate the maximum start time to ensure we don't go beyond video end
        max_start_time = video_duration - self.duration_seconds
        
        # Start from 10% of video duration to avoid beginning
        min_start_time = video_duration * 0.1
        
        # End at 90% of video duration to avoid ending
        max_start_time = min(max_start_time, video_duration * 0.9)
        
        # Generate random start time between min and max
        start_time = random.uniform(min_start_time, max_start_time)
        
        return start_time
    
    def _generate_output_filename(self, video_path, start_time):
        """
        Generate a unique output filename for the clip.
        
        Args:
            video_path (str): Original video file path
            start_time (float): Start time of the clip
            
        Returns:
            str: Output filename
        """
        # Get original filename without extension
        video_name = Path(video_path).stem
        
        # Create filename with start time
        output_filename = f"{video_name}_clip_{start_time:.1f}s.mp4"
        
        # Ensure filename is valid
        output_filename = "".join(c for c in output_filename if c.isalnum() or c in (' ', '-', '_', '.')).rstrip()
        output_filename = output_filename.replace(' ', '_')
        
        return output_filename
    
    def export_clip_from_video(self, video_path):
        """
        Export a random clip from a single video.
        
        Args:
            video_path (str): Path to the input video file
            
        Returns:
            str: Path to the exported clip, or None if failed
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
                    return None
                
                # Get random start time
                start_time = self._get_random_start_time(video_duration)
                end_time = start_time + self.duration_seconds
                
                print(f"  Extracting clip from {start_time:.2f}s to {end_time:.2f}s")
                
                # Extract subclip
                subclip = clip.subclipped(start_time, end_time)
                
                # Resize to target resolution
                subclip = subclip.resized((self.width, self.height))
                
                # Generate output filename
                output_filename = self._generate_output_filename(video_path, start_time)
                output_path = os.path.join(self.output_path, output_filename)
                
                # Export the clip
                print(f"  Exporting to: {output_path}")
                
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
                        print(f"  ✓ Successfully exported: {output_filename} ({file_size:,} bytes)")
                        return output_path
                    else:
                        print(f"  ✗ Error: File was created but is empty: {output_path}")
                        # Try to remove empty file
                        try:
                            os.remove(output_path)
                        except:
                            pass
                        return None
                else:
                    print(f"  ✗ Error: File was not saved to {output_path}")
                    return None
                
        except Exception as e:
            print(f"  ✗ Error processing video {video_path}: {e}")
            import traceback
            print(f"  Error details: {traceback.format_exc()}")
            return None
    
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
            
            exported_path = self.export_clip_from_video(video_path)
            if exported_path:
                exported_clips.append(exported_path)
        
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
            result = self.export_clip_from_video(test_video_path)
            if result:
                print(f"✓ Test successful! Clip saved to: {result}")
                return True
            else:
                print(f"✗ Test failed! No clip was saved.")
                return False
                
        except Exception as e:
            print(f"✗ Test failed with error: {e}")
            import traceback
            print(f"Error details: {traceback.format_exc()}")
            return False
