"""
Search for raw images by filename prefix.
"""

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# Prefixes extracted from annotation files
SEARCH_PREFIXES = [
    "00ca7adf",
    "01468aa0", 
    "019eea56"
]

def search_bucket():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'allencell'
    
    prefix = "aics/integrated_transcriptomics_structural_organization_hipsc_cm/"
    
    print(f"Searching for file prefixes: {SEARCH_PREFIXES} in {prefix}\n")
    
    # This prefix likely contains many files, we need to scan it.
    # Since we know the annotation is there, the raw file should be there too or in a subdirectory.
    
    paginator = s3.get_paginator('list_objects_v2')
    # Increase check limit significantly
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)

    found_count = 0
    checked_count = 0
    
    for page in page_iterator:
        if 'Contents' not in page: continue
        
        for obj in page['Contents']:
            key = obj['Key']
            checked_count += 1
            
            # Check if key starts with or contains our target prefixes
            if any(p in key for p in SEARCH_PREFIXES):
                # We want files that are NOT annotations
                if "annotation" not in key and key.endswith(('.tif', '.tiff')):
                    print(f"  FOUND RAW CANDIDATE: {key}")
                    found_count += 1
        
        # Stop after checking enough files to avoid hanging 
        if checked_count > 5000:
            print("Checked 5000 items, stopping.")
            break
            
    if found_count == 0:
        print("No matches found.")

if __name__ == "__main__":
    search_bucket()
