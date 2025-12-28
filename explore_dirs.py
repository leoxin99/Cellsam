"""
Explore directories in the dataset to find raw images.
"""
import boto3
from botocore import UNSIGNED
from botocore.config import Config

def explore_structure():
    s3 = boto3.client('s3', config=Config(signature_version=UNSIGNED))
    bucket = 'allencell'
    prefix = "aics/integrated_transcriptomics_structural_organization_hipsc_cm/"
    
    print(f"checking subdirectories in {prefix}")
    
    # Use Delimiter to get "sub-folders"
    resp = s3.list_objects_v2(Bucket=bucket, Prefix=prefix, Delimiter='/')
    
    if 'CommonPrefixes' in resp:
        print("Subdirectories:")
        for p in resp['CommonPrefixes']:
            print(f"  {p['Prefix']}")
            
            # Check one level deeper
            sub_resp = s3.list_objects_v2(Bucket=bucket, Prefix=p['Prefix'], Delimiter='/')
            if 'CommonPrefixes' in sub_resp:
                for sub_p in sub_resp['CommonPrefixes']:
                    print(f"    {sub_p['Prefix']}")

if __name__ == "__main__":
    explore_structure()
