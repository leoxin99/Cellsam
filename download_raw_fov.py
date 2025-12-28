"""
Download raw FOV images from Allen Cell pipeline_integrated_single_cell package.
"""

import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm

# S3 settings
BUCKET = "allencell"
PREFIX = "aics/pipeline_integrated_single_cell/fov_datasets/"
OUTPUT_DIR = "d:/AI/paper/CellSam/allen_raw_fov"

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
    print(f"Found {len(files)} files")
    
    # Find TIFF files
    tiff_files = [f for f in files if f['Key'].endswith(('.tif', '.tiff', '.ome.tiff'))]
    print(f"Found {len(tiff_files)} TIFF files")
    
    if not tiff_files:
        print("\nListing all file extensions found:")
        extensions = set()
        for f in files:
            ext = os.path.splitext(f['Key'])[1]
            extensions.add(ext)
        print(extensions)
        
        print("\nFirst 10 files:")
        for f in files[:10]:
            print(f"  {os.path.basename(f['Key'])} - {f['Size']//1024}KB")
        return
    
    # Download first 5 TIFF files
    print(f"\nDownloading first 5 TIFF files to {OUTPUT_DIR}")
    
    for obj in tiff_files[:5]:
        key = obj['Key']
        filename = os.path.basename(key)
        dest_path = os.path.join(OUTPUT_DIR, filename)
        
        file_size = obj['Size']
        print(f"Downloading: {filename} ({file_size//1024//1024}MB)")
        
        with tqdm(total=file_size, unit='B', unit_scale=True) as pbar:
            s3.download_file(
                BUCKET, key, dest_path,
                Callback=lambda b: pbar.update(b)
            )
    
    print(f"\nDone! Files saved to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
