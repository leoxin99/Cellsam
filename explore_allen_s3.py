"""
Explore Allen Cell S3 bucket to find raw image data.
"""

import boto3
from botocore import UNSIGNED
from botocore.config import Config

def explore_bucket():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    
    # Try different known prefixes
    prefixes_to_check = [
        "aics/",
        "aics/cardio-classifier-brightfield/",
        "aics/hipsc_single_cell_image_dataset/",
        "aics/actk/",
    ]
    
    for prefix in prefixes_to_check:
        print(f"\n=== Checking: s3://allencell/{prefix} ===")
        
        try:
            resp = s3.list_objects_v2(Bucket='allencell', Prefix=prefix, Delimiter='/', MaxKeys=30)
            
            # Show subdirectories
            if 'CommonPrefixes' in resp:
                print("Subdirectories:")
                for p in resp['CommonPrefixes'][:10]:
                    print(f"  {p['Prefix']}")
            
            # Show files
            if 'Contents' in resp:
                print(f"Files ({len(resp['Contents'])} found):")
                for obj in resp['Contents'][:5]:
                    size_mb = obj['Size'] / (1024 * 1024)
                    print(f"  {obj['Key'].split('/')[-1]} ({size_mb:.1f}MB)")
                    
        except Exception as e:
            print(f"  Error: {e}")
    
    # Specifically look at cardio-classifier-brightfield
    print("\n=== Detailed: cardio-classifier-brightfield ===")
    resp = s3.list_objects_v2(Bucket='allencell', Prefix='aics/cardio-classifier-brightfield/', MaxKeys=50)
    
    if 'Contents' in resp:
        for obj in resp['Contents'][:20]:
            size_mb = obj['Size'] / (1024 * 1024)
            filename = obj['Key'].split('/')[-1]
            print(f"  {filename} ({size_mb:.1f}MB)")

if __name__ == "__main__":
    explore_bucket()
