"""
Download cardiomyocyte brightfield images from Allen Cell.
"""

import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

# S3 settings
BUCKET = "allencell"
PREFIX = "aics/cardio-classifier-brightfield/"
OUTPUT_DIR = "d:/AI/paper/CellSam/allen_brightfield"

def main():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print(f"Listing files in s3://{BUCKET}/{PREFIX}")
    
    # List files
    resp = s3.list_objects_v2(Bucket=BUCKET, Prefix=PREFIX, MaxKeys=100)
    
    if 'Contents' not in resp:
        print("No files found at this prefix")
        return
    
    files = resp['Contents']
    print(f"Found {len(files)} files total")
    
    # Find smaller TIFF files (< 50MB for quick testing)
    tiff_files = [f for f in files if f['Key'].endswith(('.tif', '.tiff')) and f['Size'] < 50 * 1024 * 1024]
    
    if not tiff_files:
        # If no small files, take any TIFF
        tiff_files = [f for f in files if f['Key'].endswith(('.tif', '.tiff'))][:3]
    
    print(f"Selected {len(tiff_files)} TIFF files for download")
    
    # Download files
    for obj in tiff_files[:5]:
        key = obj['Key']
        filename = os.path.basename(key)
        dest_path = os.path.join(OUTPUT_DIR, filename)
        
        file_size = obj['Size']
        print(f"\nDownloading: {filename} ({file_size//1024//1024}MB)")
        
        with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
            s3.download_file(
                BUCKET, key, dest_path,
                Callback=lambda b: pbar.update(b)
            )
    
    print(f"\nDone! Files saved to {OUTPUT_DIR}")
    
    # List downloaded files
    print("\nDownloaded files:")
    for f in os.listdir(OUTPUT_DIR):
        size = os.path.getsize(os.path.join(OUTPUT_DIR, f)) / (1024*1024)
        print(f"  {f} ({size:.1f}MB)")

if __name__ == "__main__":
    main()
