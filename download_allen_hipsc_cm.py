"""
Download Allen hiPSC-CM dataset using direct S3 access.
The Allen Cell Collection is publicly accessible on S3.

Usage:
    python download_allen_hipsc_cm.py --output_dir ./allen_data --num_samples 10

Requirements:
    pip install boto3 tqdm
"""

import os
import argparse
from pathlib import Path
from typing import List, Optional

import boto3
from botocore import UNSIGNED
from botocore.config import Config
from tqdm import tqdm


# S3 bucket and prefix for Allen Cell Collection
BUCKET_NAME = "allencell"
PACKAGE_PREFIX = "aics/integrated_transcriptomics_structural_organization_hipsc_cm/"


def list_s3_files(
    bucket: str,
    prefix: str,
    max_files: Optional[int] = None,
    extensions: Optional[List[str]] = None
) -> List[str]:
    """
    List files in an S3 bucket with optional filtering.
    
    Args:
        bucket: S3 bucket name
        prefix: S3 prefix (folder path)
        max_files: Maximum number of files to return
        extensions: Filter by file extensions (e.g., ['.tif', '.tiff'])
    
    Returns:
        List of S3 keys
    """
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    files = []
    paginator = s3.get_paginator('list_objects_v2')
    
    for page in paginator.paginate(Bucket=bucket, Prefix=prefix):
        if 'Contents' not in page:
            continue
        
        for obj in page['Contents']:
            key = obj['Key']
            
            # Filter by extension if specified
            if extensions:
                if not any(key.lower().endswith(ext.lower()) for ext in extensions):
                    continue
            
            files.append(key)
            
            if max_files and len(files) >= max_files:
                return files
    
    return files


def download_file(bucket: str, key: str, dest_path: str) -> bool:
    """
    Download a single file from S3.
    
    Returns:
        True if successful, False otherwise
    """
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    os.makedirs(os.path.dirname(dest_path), exist_ok=True)
    
    try:
        # Get file size for progress bar
        response = s3.head_object(Bucket=bucket, Key=key)
        file_size = response['ContentLength']
        
        # Download with progress
        with tqdm(total=file_size, unit='B', unit_scale=True, desc=os.path.basename(key)) as pbar:
            s3.download_file(
                bucket, key, dest_path,
                Callback=lambda bytes_transferred: pbar.update(bytes_transferred)
            )
        return True
    except Exception as e:
        print(f"Error downloading {key}: {e}")
        return False


def download_dataset(
    output_dir: str,
    num_samples: Optional[int] = None,
    extensions: Optional[List[str]] = None,
    dry_run: bool = False
):
    """
    Download files from the Allen hiPSC-CM dataset.
    
    Args:
        output_dir: Directory to save files
        num_samples: Number of files to download (None = all)
        extensions: Filter by file extensions
        dry_run: List files without downloading
    """
    print(f"Scanning S3 bucket: s3://{BUCKET_NAME}/{PACKAGE_PREFIX}")
    
    # Default to image formats
    if extensions is None:
        extensions = ['.tif', '.tiff', '.ome.tiff', '.ome.tif']
    
    # List available files
    files = list_s3_files(BUCKET_NAME, PACKAGE_PREFIX, max_files=num_samples, extensions=extensions)
    
    if not files:
        print("No files found matching criteria.")
        print("\nTrying to list all files in the prefix...")
        all_files = list_s3_files(BUCKET_NAME, PACKAGE_PREFIX, max_files=50)
        if all_files:
            print(f"Found {len(all_files)} files:")
            for f in all_files[:20]:
                print(f"  {f}")
            if len(all_files) > 20:
                print(f"  ... and {len(all_files) - 20} more")
        return
    
    print(f"Found {len(files)} files matching criteria")
    
    if dry_run:
        print("\nFiles (dry run, not downloading):")
        for f in files:
            print(f"  {f}")
        return
    
    # Create output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Download files
    print(f"\nDownloading to: {output_dir}")
    downloaded = 0
    failed = 0
    
    for key in files:
        # Create local path preserving some structure
        filename = os.path.basename(key)
        dest_path = os.path.join(output_dir, filename)
        
        if download_file(BUCKET_NAME, key, dest_path):
            downloaded += 1
        else:
            failed += 1
    
    print(f"\nDownload complete!")
    print(f"  Success: {downloaded}")
    print(f"  Failed: {failed}")
    print(f"  Saved to: {output_dir}")


def explore_bucket():
    """
    Explore the S3 bucket structure to find available data.
    """
    print(f"Exploring S3 bucket: s3://{BUCKET_NAME}/")
    
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # List top-level prefixes
    paginator = s3.get_paginator('list_objects_v2')
    
    print("\nTop-level directories in 'aics/' prefix:")
    prefixes_seen = set()
    
    for page in paginator.paginate(Bucket=BUCKET_NAME, Prefix="aics/", Delimiter="/"):
        if 'CommonPrefixes' in page:
            for prefix in page['CommonPrefixes']:
                print(f"  {prefix['Prefix']}")
                prefixes_seen.add(prefix['Prefix'])
    
    # Also try listing the specific package
    print(f"\nListing files in: {PACKAGE_PREFIX}")
    files = list_s3_files(BUCKET_NAME, PACKAGE_PREFIX, max_files=30)
    
    if files:
        print(f"Found {len(files)} files:")
        for f in files[:15]:
            size_str = ""
            try:
                response = s3.head_object(Bucket=BUCKET_NAME, Key=f)
                size_mb = response['ContentLength'] / (1024 * 1024)
                size_str = f" ({size_mb:.1f} MB)"
            except:
                pass
            print(f"  {os.path.basename(f)}{size_str}")
        if len(files) > 15:
            print(f"  ... and {len(files) - 15} more files")
    else:
        print("  No files found at this prefix.")


def main():
    parser = argparse.ArgumentParser(description="Download Allen hiPSC-CM dataset from S3")
    parser.add_argument("--output_dir", type=str, default="d:/AI/paper/CellSam/allen_data",
                       help="Output directory")
    parser.add_argument("--num_samples", type=int, default=None,
                       help="Number of files to download")
    parser.add_argument("--extensions", type=str, nargs="+", default=None,
                       help="File extensions to include (e.g., .tif .tiff)")
    parser.add_argument("--dry_run", action="store_true",
                       help="List files without downloading")
    parser.add_argument("--explore", action="store_true",
                       help="Explore bucket structure")
    
    args = parser.parse_args()
    
    if args.explore:
        explore_bucket()
    else:
        download_dataset(
            args.output_dir,
            num_samples=args.num_samples,
            extensions=args.extensions,
            dry_run=args.dry_run
        )


if __name__ == "__main__":
    main()
