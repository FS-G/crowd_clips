import json
import os
from pathlib import Path
from utils.ceremony_detector import VideoLabel
from utils.video_validator import VideoValidator


def load_parameters(config_path="parameters.json"):
    """
    Load and parse the parameters from the JSON configuration file.
    
    Args:
        config_path (str): Path to the parameters JSON file
        
    Returns:
        dict: Parsed configuration parameters
    """
    try:
        with open(config_path, 'r') as file:
            config = json.load(file)
        return config
    except FileNotFoundError:
        raise FileNotFoundError(f"Configuration file '{config_path}' not found")
    except json.JSONDecodeError:
        raise ValueError(f"Invalid JSON format in '{config_path}'")


def process_videos():
    """
    Main function to process videos from all input folders.
    """
    # Load configuration
    config = load_parameters()
    
    # Extract parameters
    input_folders = config.get("in_folders", [])
    output_path = config.get("output_path", "")
    export_settings = config.get("export_settings", {})
    
    print(f"Found {len(input_folders)} input folders to process")
    print(f"Output path: {output_path}")
    print(f"Export settings: {export_settings}")
    
    # Initialize video validator
    video_validator = VideoValidator(min_duration_minutes=10)
    
    # Initialize ceremony detector
    ceremony_detector = VideoLabel()
    
    all_ceremony_videos = []
    
    # Process each input folder
    for folder_path in input_folders:
        print(f"\nProcessing folder: {folder_path}")
        
        if not os.path.exists(folder_path):
            print(f"Warning: Folder '{folder_path}' does not exist, skipping...")
            continue
        
        # Get valid videos from this folder
        valid_videos = video_validator.get_valid_videos(folder_path)
        print(f"Found {len(valid_videos)} valid videos (>= 10 minutes) in {folder_path}")
        
        if not valid_videos:
            print("No valid videos found in this folder, continuing to next...")
            continue
        
        # Process videos through ceremony detector
        ceremony_videos = ceremony_detector.process_video_list(valid_videos)
        print(f"Found {len(ceremony_videos)} ceremony videos in {folder_path}")
        
        all_ceremony_videos.extend(ceremony_videos)
    
    print(f"\nTotal ceremony videos found across all folders: {len(all_ceremony_videos)}")
    
    # Print summary of found ceremony videos
    if all_ceremony_videos:
        print("\nCeremony videos found:")
        for video_path in all_ceremony_videos:
            print(f"  - {video_path}")
    
    return all_ceremony_videos


if __name__ == "__main__":
    try:
        ceremony_videos = process_videos()
        print(f"\nProcessing complete. Found {len(ceremony_videos)} ceremony videos.")
    except Exception as e:
        print(f"Error during processing: {e}")
        raise
