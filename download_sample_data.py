import os
import requests
import zipfile
import io

def download_and_extract_sample(url, target_dir):
    print(f"Downloading sample dataset from {url}...")
    try:
        response = requests.get(url)
        response.raise_for_status()
        
        print("Download complete. Extracting...")
        with zipfile.ZipFile(io.BytesIO(response.content)) as z:
            z.extractall(target_dir)
            
        print(f"Success! Data extracted to: {target_dir}")
        print("You can now drag the '01' or '02' folder from inside that directory into Napari.")
        
    except Exception as e:
        print(f"Error downloading: {e}")

if __name__ == "__main__":
    # Using PhC-C2DH-U373 (Glioblastoma-astrocytoma) as it's a small (40MB) Phase Contrast dataset
    # This is excellent for testing Cell-SAM.
    DATASET_URL = "https://data.celltrackingchallenge.net/training-datasets/PhC-C2DH-U373.zip"
    TARGET_DIR = "d:\\AI\\paper\\CellSam\\sample_data"
    
    if not os.path.exists(TARGET_DIR):
        os.makedirs(TARGET_DIR)
        
    download_and_extract_sample(DATASET_URL, TARGET_DIR)
