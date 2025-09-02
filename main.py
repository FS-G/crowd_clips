import json
import os
from pathlib import Path
from utils.ceremony_detector import VideoLabel
from utils.video_validator import VideoValidator
from utils.crowd_clip_exporter import CrowdClipExporter


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
    
    # Initialize clip exporter
    resolution = export_settings.get("resolution", "1920x1080")
    duration_needed_seconds = export_settings.get("duration_needed_seconds", 5)
    clip_exporter = CrowdClipExporter(
        output_path=output_path,
        resolution=resolution,
        duration_seconds=duration_needed_seconds
    )
    
    # Test the clip exporter with a sample video if available
    print(f"\nTesting clip exporter...")
    test_video_found = False
    for folder_path in input_folders:
        if os.path.exists(folder_path):
            # Look for any video file to test with
            for root, dirs, files in os.walk(folder_path):
                for file in files:
                    if file.lower().endswith(('.mp4', '.avi', '.mov', '.mkv')):
                        test_video_path = os.path.join(root, file)
                        print(f"Found test video: {test_video_path}")
                        test_result = clip_exporter.test_video_processing(test_video_path)
                        if test_result:
                            print("✓ Clip exporter test successful!")
                            test_video_found = True
                        else:
                            print("✗ Clip exporter test failed!")
                        break
                if test_video_found:
                    break
        if test_video_found:
            break
    
    if not test_video_found:
        print("⚠ No test video found for testing clip exporter")
    
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
        
        # Export clips from ceremony videos
        print(f"\n{'='*60}")
        print("EXPORTING CLIPS FROM CEREMONY VIDEOS")
        print(f"{'='*60}")
        
        exported_clips = clip_exporter.export_clips_from_videos(all_ceremony_videos)
        
        print(f"\n{'='*60}")
        print("PROCESSING SUMMARY")
        print(f"{'='*60}")
        print(f"Total ceremony videos processed: {len(all_ceremony_videos)}")
        print(f"Total clips exported: {len(exported_clips)}")
        print(f"Output directory: {output_path}")
        print(f"Clip resolution: {resolution}")
        print(f"Clip duration: {duration_needed_seconds} seconds")
        
        # Verify output directory contents
        if exported_clips:
            print(f"\nOutput directory contents:")
            try:
                output_files = os.listdir(output_path)
                mp4_files = [f for f in output_files if f.endswith('.mp4')]
                print(f"  - Total files in output directory: {len(output_files)}")
                print(f"  - MP4 clip files: {len(mp4_files)}")
                if mp4_files:
                    print(f"  - Sample clip files:")
                    for i, file in enumerate(mp4_files[:5]):  # Show first 5 files
                        file_path = os.path.join(output_path, file)
                        file_size = os.path.getsize(file_path)
                        print(f"    {i+1}. {file} ({file_size:,} bytes)")
                    if len(mp4_files) > 5:
                        print(f"    ... and {len(mp4_files) - 5} more files")
            except Exception as e:
                print(f"  - Error reading output directory: {e}")
        
    else:
        print("No ceremony videos found to export clips from.")
    
    return all_ceremony_videos


if __name__ == "__main__":
    try:
        ceremony_videos = process_videos()
        print(f"\nProcessing complete. Found {len(ceremony_videos)} ceremony videos.")
    except Exception as e:
        print(f"Error during processing: {e}")
        raise
