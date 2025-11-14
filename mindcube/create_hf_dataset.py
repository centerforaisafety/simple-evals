#!/usr/bin/env python3
import json
import base64
import os
from pathlib import Path
from datasets import Dataset
from dotenv import load_dotenv
from tqdm import tqdm
import argparse

# Load environment variables
load_dotenv()

def load_mindcube_raw():
    """Load the raw MindCube-Tiny dataset."""
    import requests
    import zipfile
    import tempfile
    
    data_dir = Path("mindcube/data")
    
    # Check if data already exists
    if (data_dir / "raw" / "MindCube_tinybench.jsonl").exists():
        print(f"✅ MindCube-Tiny dataset already exists at {data_dir}")
    else:
        print("📥 Downloading MindCube dataset...")
        data_dir.mkdir(parents=True, exist_ok=True)
        
        # Download URL from HuggingFace
        repo_url = "https://huggingface.co/datasets/Inevitablevalor/MindCube/resolve/main/data.zip"
        
        try:
            # Download the zip file
            response = requests.get(repo_url, stream=True)
            response.raise_for_status()
            
            # Save to temporary file
            with tempfile.NamedTemporaryFile(delete=False, suffix='.zip') as tmp_file:
                for chunk in response.iter_content(chunk_size=8192):
                    tmp_file.write(chunk)
                temp_zip_path = tmp_file.name
            
            # Extract the zip file
            print("📦 Extracting dataset...")
            with zipfile.ZipFile(temp_zip_path, 'r') as zip_ref:
                zip_ref.extractall(data_dir.parent)
            
            # Clean up temporary file
            os.unlink(temp_zip_path)
            print(f"✅ Dataset extracted to {data_dir}")
            
        except requests.RequestException as e:
            print(f"❌ Failed to download dataset: {e}")
            raise
    
    # Load the dataset
    tinybench_path = data_dir / "raw" / "MindCube_tinybench.jsonl"
    
    print(f"Loading MindCube-Tiny dataset from {tinybench_path}")
    examples = []
    with open(tinybench_path, 'r') as f:
        for line in f:
            if line.strip():
                examples.append(json.loads(line.strip()))
    
    print(f"Loaded {len(examples)} examples from MindCube-Tiny")
    return examples, data_dir

def check_example_validity(example, data_dir):
    """Check if an example has all required images."""
    images = example.get('images', [])
    missing_images = []
    
    for img_path in images:
        full_img_path = data_dir / img_path
        if not full_img_path.exists():
            missing_images.append(str(full_img_path))
    
    return len(missing_images) == 0

def convert_example_to_hf_format(example, data_dir):
    """Convert a MindCube example to Hugging Face dataset format."""
    # Extract basic info
    question = example.get('question', '')
    answer = example.get('gt_answer', '')
    example_id = example.get('id', '')
    category = example.get('category', [])
    question_type = example.get('type', '')
    meta_info = example.get('meta_info', [])
    
    # Convert images to base64
    images = example.get('images', [])
    base64_images = []
    
    for img_path in images:
        full_img_path = data_dir / img_path
        with open(full_img_path, 'rb') as f:
            img_data = f.read()
            img_b64 = base64.b64encode(img_data).decode('utf-8')
            base64_images.append(img_b64)
    
    # Convert complex structures to strings to avoid Arrow type issues
    meta_info_str = json.dumps(meta_info) if meta_info else ""
    category_str = json.dumps(category) if category else ""
    image_paths_str = json.dumps(images) if images else ""
    
    # Extract setting from ID (matching get_setting_from_id logic)
    def get_setting_from_id(item_id: str) -> str:
        if not item_id:
            return 'other'
        
        item_id_lower = item_id.lower()
        
        if 'around' in item_id_lower:
            return 'around'
        elif 'rotation' in item_id_lower:
            return 'rotation'
        elif 'translation' in item_id_lower:
            return 'translation'
        elif 'among' in item_id_lower:
            return 'among'
        else:
            return 'other'
    
    setting = get_setting_from_id(example_id)
    
    # Ensure all fields are consistent types (strings)
    return {
        'id': str(example_id),
        'question': str(question),
        'answer': str(answer),
        'category': category_str,  # Store as JSON string
        'question_type': str(question_type),  # Ensure string type
        'meta_info': meta_info_str,  # Store as JSON string
        'images_base64': base64_images,  # List of strings
        'num_images': int(len(base64_images)),  # Ensure int type
        'original_image_paths': image_paths_str,  # Store as JSON string
        'setting': str(setting)  # Add setting field
    }

def create_mindcube_hf_dataset(output_name="justinphan3110/mindcube", private=True):
    """Create and upload MindCube dataset to Hugging Face."""
    
    # Load raw examples
    print("Loading raw MindCube dataset...")
    examples, data_dir = load_mindcube_raw()
    
    # Filter valid examples and convert to HF format
    print("Filtering valid examples and converting to HF format...")
    valid_examples = []
    skipped_count = 0
    
    for i, example in enumerate(tqdm(examples, desc="Processing examples")):
        if check_example_validity(example, data_dir):
            try:
                hf_example = convert_example_to_hf_format(example, data_dir)
                
                # Test if this example can be converted to dataset format
                try:
                    from datasets import Dataset
                    test_dataset = Dataset.from_list([hf_example])
                    valid_examples.append(hf_example)
                except Exception as dataset_error:
                    print(f"Example {i} ({example.get('id', 'unknown')}) failed dataset conversion: {dataset_error}")
                    skipped_count += 1
                    
            except Exception as e:
                print(f"Error converting example {i}: {e}")
                skipped_count += 1
        else:
            skipped_count += 1
    
    print(f"✅ Processed {len(valid_examples)} valid examples")
    print(f"⚠️ Skipped {skipped_count} examples due to missing images or errors")
    
    # Create Hugging Face dataset in batches to avoid memory issues
    print("Creating Hugging Face dataset...")
    
    # Try to create dataset with explicit schema to avoid type inference issues
    try:
        # Process in smaller batches first to test
        batch_size = 100
        print(f"Testing with first {batch_size} examples...")
        test_batch = valid_examples[:batch_size]
        test_dataset = Dataset.from_list(test_batch)
        print(f"✅ Test batch successful with {len(test_dataset)} examples")
        
        # If test succeeds, try full dataset
        print("Creating full dataset...")
        dataset = Dataset.from_list(valid_examples)
        
    except Exception as e:
        print(f"❌ Dataset creation failed: {e}")
        print("Trying alternative approach with explicit typing...")
        
        # Alternative: save to JSON and load back
        import tempfile
        with tempfile.NamedTemporaryFile(mode='w', suffix='.jsonl', delete=False) as f:
            for example in valid_examples:
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
    print(f"  • Total examples: {len(valid_examples)}")
    print(f"  • Question types: {set(ex['question_type'] for ex in valid_examples)}")
    print(f"  • Categories: {set(str(ex['category']) for ex in valid_examples)}")
    
    
    return dataset

def main():
    parser = argparse.ArgumentParser(description="Convert MindCube dataset to Hugging Face format")
    parser.add_argument("--output_name",
                       help="Hugging Face dataset name")
    parser.add_argument("--public", action="store_true", 
                       help="Make dataset public (default: private)")
    
    args = parser.parse_args()
    
    create_mindcube_hf_dataset(
        output_name=args.output_name,
        private=not args.public
    )

if __name__ == '__main__':
    main()