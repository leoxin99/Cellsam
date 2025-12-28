"""
Search for our raw image ID in the integrated transcriptomics dataset.
"""
import boto3
from botocore import UNSIGNED
from botocore.config import Config

SEARCH_ID = "5500000101"

def search_id():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'allencell'
    prefix = "aics/integrated_transcriptomics_structural_organization_hipsc_cm/"
    
    print(f"Searching for {SEARCH_ID} in {prefix}")
    
    paginator = s3.get_paginator('list_objects_v2')
    page_iterator = paginator.paginate(Bucket=bucket, Prefix=prefix)
    
    found = False
    count = 0
    for page in page_iterator:
        if 'Contents' not in page: continue
        for obj in page['Contents']:
            count += 1
            if SEARCH_ID in obj['Key']:
                print(f"FOUND MATCH: {obj['Key']}")
                found = True
        
        if count > 10000:
            print("Checked 10000 items, stopping.")
            break
            
    if not found:
        print("No match found.")

if __name__ == "__main__":
    search_id()
