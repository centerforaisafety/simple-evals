#!/usr/bin/env python3
"""
IntPhys2 Dataset Converter
Converts IntPhys2 dataset to Hugging Face dataset format.
"""

import json
import base64
import os
import cv2
import pandas as pd
from pathlib import Path
from datasets import Dataset
from dotenv import load_dotenv
from tqdm import tqdm
import argparse
import requests
import zipfile
import tempfile

# Load environment variables
load_dotenv()

def download_and_extract_intphys2():
    """Download and extract IntPhys2 dataset if not present."""
    data_dir = Path("intphys2/data")
    main_dir = data_dir / "Main"
    
    if main_dir.exists():
        print(f"✅ Dataset already exists at {main_dir}")
        return str(data_dir)
    
    print("📥 Downloading IntPhys2 dataset...")
    data_dir.mkdir(parents=True, exist_ok=True)
    
    # Download URL
    download_url = "https://dl.fbaipublicfiles.com/IntPhys2/IntPhys2.zip"
    
    try:
        # Download the zip file
        response = requests.get(download_url, stream=True)
        response.raise_for_status()
        
        # Save to temporary file
        with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
            for chunk in response.iter_content(chunk_size=8192):
                tmp_file.write(chunk)
            temp_zip_path = tmp_file.name
        
        # Extract the zip file
        print("📦 Extracting dataset...")
        with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
            zip_ref.extractall(data_dir)
        
        # Clean up temporary file
        os.unlink(temp_zip_path)
        print(f"✅ Dataset extracted to {data_dir}")
        
    except requests.RequestException as e:
        print(f"❌ Failed to download dataset: {e}")
        raise
    
    return str(data_dir)

def process_video(video_path, seconds_per_frame=1.5, max_frames=50):
    """Extracts video frames at specified time intervals and encodes them as base64.
    
    Args:
        video_path (str): Path to input video file
        seconds_per_frame (float): Time interval between frame captures in seconds
        max_frames (int): Maximum number of frames to extract
        
    Returns:
        list: Base64-encoded JPEG frames of the video
    """
    base64Frames = []
    
    # Initialize video capture with OpenCV
    video = cv2.VideoCapture(video_path)
    total_frames = int(video.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = video.get(cv2.CAP_PROP_FPS)
    frames_to_skip = int(fps * seconds_per_frame)  # Calculate frame skip interval
    curr_frame = 0

    # Frame extraction loop
    while curr_frame < total_frames - 1 and len(base64Frames) < max_frames:
        video.set(cv2.CAP_PROP_POS_FRAMES, curr_frame)
        success, frame = video.read()
        if not success:
            break
        # Convert frame to JPEG and base64 encode
        _, buffer = cv2.imencode(".jpg", frame)
        base64Frames.append(base64.b64encode(buffer).decode("utf-8"))
        curr_frame += frames_to_skip  # Skip frames based on interval
    
    video.release()
    return base64Frames

def convert_scene_to_hf_format(scene_idx, df_metadata, data_folder):
    """Convert a single IntPhys2 scene to Hugging Face dataset format."""
    scene_data = df_metadata[df_metadata["SceneIndex"] == scene_idx]
    
    if len(scene_data) != 4:
        print(f"Warning: Scene {scene_idx} doesn't have exactly 4 videos, skipping")
        return []
    
    split = "Main"
    videos_dir = Path(data_folder) / split / "Videos"
    
    # Process all 4 video scenarios for this scene
    hf_examples = []
    
    for _, row in scene_data.iterrows():
        video_name = row['name']
        video_type = row['type']
        video_path = videos_dir / f"{video_name}.mp4"
        
        if not video_path.exists():
            print(f"Warning: Video {video_path} not found, skipping")
            continue
        
        # Extract frames from video
        try:
            base64_frames = process_video(str(video_path), seconds_per_frame=1, max_frames=50)
            
            # Determine ground truth from type
            if "Possible" in video_type:
                ground_truth = 1  # Plausible
                is_plausible = True
            elif "Impossible" in video_type:
                ground_truth = 0  # Implausible
                is_plausible = False
            else:
                print(f"Warning: Unknown video type {video_type}, skipping")
                continue
            
            # Create HF example
            hf_example = {
                'id': f"scene_{scene_idx}_{video_name}",
                'scene_idx': int(scene_idx),
                'video_name': str(video_name),
                'video_type': str(video_type),
                'ground_truth': int(ground_truth),
                'is_plausible': bool(is_plausible),
                'frames_base64': base64_frames,
                'num_frames': int(len(base64_frames)),
                'original_video_path': str(video_path.relative_to(Path(data_folder)))
            }
            
            hf_examples.append(hf_example)
            
        except Exception as e:
            print(f"Error processing video {video_path}: {e}")
            continue
    
    return hf_examples

def create_intphys2_hf_dataset(output_name="justinphan3110/intphys2", private=True):
    """Create and upload IntPhys2 dataset to Hugging Face."""
    
    # Download and extract raw dataset
    print("Downloading and extracting IntPhys2 dataset...")
    data_folder = download_and_extract_intphys2()
    
    # Load metadata
    split = "Main"
    metadata_path = Path(data_folder) / split / "metadata.csv"
    
    if not metadata_path.exists():
        raise FileNotFoundError(f"Metadata file not found: {metadata_path}")
    
    df_metadata = pd.read_csv(metadata_path)
    scene_indices = df_metadata["SceneIndex"].unique()
    
    print(f"Found {len(scene_indices)} scenes in the dataset")
    
    # Convert scenes to HF format
    print("Converting scenes to HF format...")
    all_examples = []
    skipped_count = 0
    
    for scene_idx in tqdm(scene_indices, desc="Processing scenes"):
        try:
            scene_examples = convert_scene_to_hf_format(scene_idx, df_metadata, data_folder)
            
            # Test if these examples can be converted to dataset format
            if scene_examples:
                try:
                    test_dataset = Dataset.from_list(scene_examples[:1])  # Test with one example
                    all_examples.extend(scene_examples)
                except Exception as dataset_error:
                    print(f"Scene {scene_idx} failed dataset conversion: {dataset_error}")
                    skipped_count += len(scene_examples)
            else:
                skipped_count += 4  # Each scene should have 4 videos
                
        except Exception as e:
            print(f"Error processing scene {scene_idx}: {e}")
            skipped_count += 4
    
    print(f"✅ Processed {len(all_examples)} valid examples")
    print(f"⚠️ Skipped {skipped_count} videos due to errors")
    
    # Create Hugging Face dataset
    print("Creating Hugging Face dataset...")
    
    try:
        # Test with a small batch first
        batch_size = 10
        print(f"Testing with first {batch_size} examples...")
        test_batch = all_examples[:batch_size]
        test_dataset = Dataset.from_list(test_batch)
        print(f"✅ Test batch successful with {len(test_dataset)} examples")
        
        # Create full dataset
        print("Creating full dataset...")
        dataset = Dataset.from_list(all_examples)
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        print("Trying alternative approach with JSON...")
        
        # Alternative: save to JSON and load back
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for example in all_examples:
                f.write(json.dumps(example) + '\n')
            temp_file = f.name
        
        try:
            dataset = Dataset.from_json(temp_file)
            os.unlink(temp_file)  # Clean up
        except Exception as json_error:
            print(f"❌ JSON approach also failed: {json_error}")
            os.unlink(temp_file)  # Clean up
            raise
    
    # Get HF token from environment
    hf_token = os.getenv('HF_TOKEN')
    if not hf_token:
        raise ValueError("HF_TOKEN not found in environment variables. Please set it in your .env file.")
    
    # Push to Hugging Face Hub
    print(f"Pushing dataset to {output_name} (private={private})...")
    dataset.push_to_hub(
        output_name,
        token=hf_token,
        private=private
    )
    
    print(f"✅ Dataset successfully uploaded to {output_name}")
    print(f"📊 Dataset statistics:")
    print(f"  • Total examples: {len(all_examples)}")
    print(f"  • Scenes: {len(set(ex['scene_idx'] for ex in all_examples))}")
    print(f"  • Video types: {set(ex['video_type'] for ex in all_examples)}")
    print(f"  • Plausible videos: {sum(1 for ex in all_examples if ex['is_plausible'])}")
    print(f"  • Implausible videos: {sum(1 for ex in all_examples if not ex['is_plausible'])}")
    
    # Print sample
    if all_examples:
        print(f"\n📝 Sample example:")
        sample = all_examples[0]
        print(f"  • ID: {sample['id']}")
        print(f"  • Scene: {sample['scene_idx']}")
        print(f"  • Video: {sample['video_name']}")
        print(f"  • Type: {sample['video_type']}")
        print(f"  • Ground truth: {sample['ground_truth']} ({'Plausible' if sample['is_plausible'] else 'Implausible'})")
        print(f"  • Frames: {sample['num_frames']}")
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Convert IntPhys2 dataset to Hugging Face format")
    parser.add_argument("--output_name", default="justinphan3110/intphys2", 
                       help="Hugging Face dataset name (default: justinphan3110/intphys2)")
    parser.add_argument("--public", action="store_true", 
                       help="Make dataset public (default: private)")
    
    args = parser.parse_args()
    
    create_intphys2_hf_dataset(
        output_name=args.output_name,
        private=not args.public
    )

if __name__ == '__main__':
    main()