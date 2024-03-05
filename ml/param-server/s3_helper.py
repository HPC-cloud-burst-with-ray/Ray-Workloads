import boto3
import os
from boto3.s3.transfer import TransferConfig
def get_s3_size(bucket_name,object_key):
    # Initialize the S3 client
    s3 = boto3.client('s3')
    s3_r=boto3.resource('s3')

    total_size=0
    bucket = s3_r.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=object_key):

        # Get the object metadata
        try:
            response = s3.head_object(Bucket=bucket_name, Key=obj.key)
    
            # Extract and print the object size
            object_size = response['ContentLength']
            total_size+=object_size
            # print("correct object",obj.key)
        except:
            print("wrong object",obj.key)
    print("Total size:", total_size, "bytes")
    return total_size
def download_s3_folder(bucket_name, s3_folder='', local_dir=None,node_type=1):
    
    bandwidth = {'0': None, '1': 8000000, '2': None}
    band_width=bandwidth[node_type]
    config=TransferConfig( max_bandwidth=band_width)
    s3=boto3.resource('s3')
    bucket = s3.Bucket(bucket_name)
    for obj in bucket.objects.filter(Prefix=s3_folder):
        
        if local_dir is None:
            target = obj.key
        elif local_dir[-1]=="/":
            target=os.path.join(local_dir, os.path.relpath(obj.key, s3_folder))
        else:
            target=local_dir

        if '/' in target:
            if not os.path.exists(os.path.dirname(target)):
                os.makedirs(os.path.dirname(target))
            
        if obj.key[-1] == '/':
            continue
        bucket.download_file(obj.key, target,Config=config)
node_type=os.getenv('LOCAL_NODE_TYPE')

DATA_DIR_S3_BASE = os.getcwd() + "/dataset_batch_s3/"
download_s3_folder("huilibucket","dataset_batch/1/",DATA_DIR_S3_BASE+'/1/',node_type)