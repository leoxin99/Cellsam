"""
Search for segmentation files in Allen Cell S3 bucket corresponding to our raw images.
"""

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# The ID of the file we downloaded: 5500000101_63x_20190610_Capture1-Capture1-B4[144]Montage-5_brf.tif
# The core ID seems to be "5500000101" or the filename without "_brf.tif"

SEARCH_IDS = [
    "5500000101",
    "5500000116"
]

def search_bucket():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'allencell'
    
    # Prefixes to search in
    prefixes = [
        "aics/imreg/segmentation/",
        "aics/pipeline_integrated_single_cell/",
        "aics/integrated_transcriptomics_structural_organization_hipsc_cm/",
        "aics/classic_features/"
    ]
    
    print(f"Searching for IDs: {SEARCH_IDS}\n")
    
    for prefix in prefixes:
        print(f"--- Checking Prefix: {prefix} ---")
        try:
            # List objects recursively (simulated by not using Delimiter, but capped)
            # Since some directories are huge, we might not find them easily by listing all.
            # let's try to list and filter
            
            paginator = s3.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix, PaginationConfig={'MaxItems': 2000})

            found_count = 0
            for page in page_iterator:
                if 'Contents' not in page:
                    continue
                    
                for obj in page['Contents']:
                    key = obj['Key']
                    # Check if any ID is in the key AND it's likely a segmentation file
                    if any(uid in key for uid in SEARCH_IDS):
                        print(f"FOUND: {key}")
                        found_count += 1
            
            if found_count == 0:
                print("No matches found in first 2000 items.")
                
        except Exception as e:
            print(f"Error searching {prefix}: {e}")

if __name__ == "__main__":
    search_bucket()
