import os
from pathlib import Path
from moviepy import VideoFileClip


class VideoValidator:
    def __init__(self, min_duration_minutes=10):
        """
        Initialize the VideoValidator class.
        
        Args:
            min_duration_minutes (int): Minimum duration in minutes for a video to be considered valid
        """
        self.min_duration_minutes = min_duration_minutes
        self.min_duration_seconds = min_duration_minutes * 60
        self.supported_extensions = {'.mp4', '.avi', '.mov', '.mkv', '.wmv', '.flv', '.webm', '.m4v'}
    
    def is_video_file(self, file_path):
        """
        Check if a file is a supported video file.
        
        Args:
            file_path (str): Path to the file
            
        Returns:
            bool: True if the file is a supported video format
        """
        return Path(file_path).suffix.lower() in self.supported_extensions
    
    def get_video_duration(self, video_path):
        """
        Get the duration of a video file in seconds.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            float: Duration in seconds, or None if error occurs
        """
        try:
            with VideoFileClip(video_path) as clip:
                return clip.duration
        except Exception as e:
            print(f"Error reading video duration for {video_path}: {e}")
            return None
    
    def is_valid_video(self, video_path):
        """
        Check if a video file meets the minimum duration requirement.
        
        Args:
            video_path (str): Path to the video file
            
        Returns:
            bool: True if the video is valid and meets duration requirements
        """
        if not self.is_video_file(video_path):
            return False
        
        if not os.path.exists(video_path):
            return False
        
        duration = self.get_video_duration(video_path)
        if duration is None:
            return False
        
        return duration >= self.min_duration_seconds
    
    def scan_directory_recursively(self, directory_path):
        """
        Recursively scan a directory for video files.
        
        Args:
            directory_path (str): Path to the directory to scan
            
        Returns:
            list: List of video file paths found in the directory and subdirectories
        """
        video_files = []
        
        try:
            for root, dirs, files in os.walk(directory_path):
                for file in files:
                    file_path = os.path.join(root, file)
                    if self.is_video_file(file_path):
                        video_files.append(file_path)
        except Exception as e:
            print(f"Error scanning directory {directory_path}: {e}")
        
        return video_files
    
    def get_valid_videos(self, directory_path):
        """
        Get all valid videos from a directory and its subdirectories.
        
        Args:
            directory_path (str): Path to the directory to scan
            
        Returns:
            list: List of valid video file paths that meet the duration requirement
        """
        print(f"Scanning directory: {directory_path}")
        
        # Get all video files recursively
        all_video_files = self.scan_directory_recursively(directory_path)
        print(f"Found {len(all_video_files)} video files in {directory_path}")
        
        # Filter for valid videos
        valid_videos = []
        for video_path in all_video_files:
            if self.is_valid_video(video_path):
                valid_videos.append(video_path)
                print(f"Valid video found: {video_path}")
            else:
                print(f"Skipping invalid video: {video_path}")
        
        return valid_videos
