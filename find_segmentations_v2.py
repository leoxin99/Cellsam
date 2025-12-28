"""
Refined search for segmentation files in Allen Cell S3 bucket.
"""

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# 5500000101
SEARCH_ID = "5500000101"

def search_bucket():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'allencell'
    
    # 1. Check the source directory again for any sibling files
    prefix_source = "aics/cardio-classifier-brightfield/"
    print(f"--- Listing {prefix_source} ---")
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix_source, MaxKeys=200)
    
    if 'Contents' in resp:
        for obj in resp['Contents']:
            key = obj['Key']
            if SEARCH_ID in key:
                print(f"  Match: {key}")
                
    # 2. Check for a segmentation folder at higher levels
    prefixes = [
        "aics/segmentation/", 
        "aics/masks/",
        "aics/cardio-classifier-masks/",
        "aics/cardio-classifier-segmentation/"
    ]
    
    for p in prefixes:
        print(f"\n--- Checking {p} ---")
        try:
            resp = s3.list_objects_v2(Bucket=bucket, Prefix=p, MaxKeys=50)
            if 'Contents' in resp:
                print(f"  Found files in {p}, checking for ID...")
                for obj in resp['Contents']:
                    if SEARCH_ID in obj['Key']:
                        print(f"  FOUND SEGMENTATION: {obj['Key']}")
            else:
                print("  No files found.")
        except:
            print("  Prefix not found.")

if __name__ == "__main__":
    search_bucket()
