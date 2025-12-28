"""
Search for raw images corresponding to existing annotations.
"""

import boto3
from botocore import UNSIGNED
from botocore.config import Config

# IDs extracted from the annotation filenames
SEARCH_IDS = [
    "5500000014",
    "5500000013"
]

def search_bucket():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'allencell'
    
    # Prefixes to search
    prefixes = [
        "aics/pipeline_integrated_single_cell/",
        "aics/cardio-classifier-brightfield/",
        "aics/integrated_transcriptomics_structural_organization_hipsc_cm/"
    ]
    
    print(f"Searching for IDs: {SEARCH_IDS}\n")
    
    for prefix in prefixes:
        print(f"--- Checking {prefix} ---")
        try:
            # We use a simulated recursive search by listing enough items
            # Or we can just filter if the prefix isn't too huge
            # For efficiency, let's just list the first few pages? No, key might be anywhere.
            # But specific collection prefixes usually group files.
            
            # For integrated_transcriptomics..., the files might be close to the annotations
            
            paginator = s3.get_paginator('list_objects_v2')
            page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix, PaginationConfig={'MaxItems': 1000})

            found = False
            for page in page_iterator:
                if 'Contents' not in page: continue
                for obj in page['Contents']:
                    if any(uid in obj['Key'] for uid in SEARCH_IDS):
                        # Filter for likely raw files (tiff, not annotation)
                        if "annotation" not in obj['Key'] and obj['Key'].endswith(('.tif', '.tiff')):
                            print(f"  FOUND RAW CANDIDATE: {obj['Key']}")
                            found = True
            
            if not found:
                print("  No matches found in first 1000 items.")

        except Exception as e:
            print(f"  Error: {e}")

if __name__ == "__main__":
    search_bucket()
