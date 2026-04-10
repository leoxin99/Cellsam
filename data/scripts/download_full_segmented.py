"""
Download FULL Allen 2D Segmented Fields dataset (all 478 images).
Uses boto3 for reliable S3 access.
"""

import os
import boto3
from botocore import UNSIGNED
from botocore.config import Config
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed


BUCKET = "allencell"
PREFIX = "aics/integrated_transcriptomics_structural_organization_hipsc_cm/2d_segmented_fields/"
PROJECT_ROOT = Path(__file__).resolve().parents[2]
OUTPUT_DIR = PROJECT_ROOT / "data" / "raw" / "allen_segmented_fields_full"


def list_all_files():
    """List all TIFF files in the 2d_segmented_fields folder."""
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    files = []
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=BUCKET, Prefix=PREFIX):
        if 'Contents' in page:
            for obj in page['Contents']:
                key = obj['Key']
                if key.endswith('.tiff') or key.endswith('.tif'):
                    files.append({
                        'key': key,
                        'size': obj['Size'],
                        'name': os.path.basename(key)
                    })
    
    return files


def download_file(file_info, output_dir):
    """Download a single file."""
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    dest_path = output_dir / file_info['name']
    
    if dest_path.exists() and dest_path.stat().st_size == file_info['size']:
        return True, file_info['name'], "skipped (exists)"
    
    try:
        s3.download_file(BUCKET, file_info['key'], str(dest_path))
        return True, file_info['name'], "downloaded"
    except Exception as e:
        return False, file_info['name'], str(e)


def main():
    print("=" * 70)
    print("Allen 2D Segmented Fields - FULL Dataset Download")
    print("=" * 70)
    
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    # List all files
    print("\nScanning S3 bucket...")
    files = list_all_files()
    
    print(f"\n📊 Dataset Statistics:")
    print(f"   Total files: {len(files)}")
    total_size = sum(f['size'] for f in files) / (1024**3)
    print(f"   Total size: {total_size:.2f} GB")
    print(f"   Output dir: {OUTPUT_DIR}")
    
    # Check existing files
    existing = [f for f in files if (OUTPUT_DIR / f['name']).exists()]
    to_download = [f for f in files if not (OUTPUT_DIR / f['name']).exists()]
    
    print(f"\n   Already downloaded: {len(existing)}")
    print(f"   To download: {len(to_download)}")
    
    if not to_download:
        print("\n✅ All files already downloaded!")
        return
    
    # Download with progress bar
    print(f"\n📥 Downloading {len(to_download)} files...")
    
    success = 0
    failed = 0
    
    for file_info in tqdm(to_download, desc="Downloading"):
        ok, name, status = download_file(file_info, OUTPUT_DIR)
        if ok:
            success += 1
        else:
            failed += 1
            print(f"\n   ❌ Failed: {name[:40]}... - {status}")
    
    print("\n" + "=" * 70)
    print("Download Complete!")
    print("=" * 70)
    print(f"   ✅ Success: {success + len(existing)}")
    print(f"   ❌ Failed: {failed}")
    print(f"   📂 Location: {OUTPUT_DIR}")
    print("=" * 70)


if __name__ == "__main__":
    main()
