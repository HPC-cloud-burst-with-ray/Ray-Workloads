import os
import sys
import time
import logging
import random
from PIL import Image, ImageFilter
from typing import Tuple, List
import glob
import numpy as np
import ray
import torch
import boto3
from torchvision import transforms as T
from boto3.s3.transfer import TransferConfig
Image.MAX_IMAGE_PIXELS = None
MAX_BANDWIDTH_HPC=94371840
use_scheduler = False
use_s3=False
s3_bucket_name=''
s3_object_name=''
# if len(sys.argv) > 1:
#     mode = sys.argv[1]
#     if mode == "sched":
#         use_scheduler = True
#         print("Using scheduler")
#     else:
#         print(f"Unknown mode: {mode}. Exiting")
#         sys.exit(1)
# if len(sys.argv) > 1:
#     mode = sys.argv[1]
#     if len(sys.argv) == 4:
#         s3_bucket_name = sys.argv[2]
#         s3_object_name = sys.argv[3]
#         use_s3 = True
#     if mode == "sched":
#         use_scheduler = True
#         print("Using scheduler")
#     else:
#         print(f"Unknown mode: {mode}. Exiting")
#         sys.exit(1)

if len(sys.argv) > 1:
    mode = sys.argv[1]
    if len(sys.argv) == 4:
        s3_bucket_name = sys.argv[2]
        s3_object_name = sys.argv[3]
        use_s3 = True
    if mode == "sched":
        use_scheduler = True
        print("Using scheduler")
    elif mode=="manu":
        use_scheduler = False
        print("Using scheduler")
    else:
        print(f"Unknown mode: {mode}. Exiting")
        sys.exit(1)
# print(use_s3,use_scheduler,s3_bucket_name,s3_object_name)
ray.init(address='auto')

DATA_DIR = os.getcwd() + "/task_images"
BATCHES = [10, 20, 30, 40, 50]
SERIAL_BATCH_TIMES = []
DISTRIBUTED_BATCH_TIMES = []
THUMB_SIZE = (64, 64)
image_list = glob.glob(DATA_DIR+"/*.jpg")
NODE_USER_NAME = "ec2-user"
DATA_IP= "10.0.0.132"


def download_s3_folder(bucket_name, s3_folder='', local_dir=None,node_type=1):
    
    bandwidth = {'0': None, '1': MAX_BANDWIDTH_HPC, '2': None}
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

def transform_image(img: object, fetch_image=True, verbose=False):
    """
    This is a deliberate compute intensive image transfromation and tensor operation
    to simulate a compute intensive image processing
    """

    before_shape = img.size

    # Make the image blur with specified intensify
    # Use torchvision transformation to augment the image
    img = img.filter(ImageFilter.GaussianBlur(radius=20))
    augmentor = T.TrivialAugmentWide(num_magnitude_bins=31)
    img = augmentor(img)

    # Convert image to tensor and transpose
    tensor = torch.tensor(np.asarray(img))
    t_tensor = torch.transpose(tensor, 0, 1)

    # compute intensive operations on tensors
    random.seed(42)
    for _ in range(30):
        tensor.pow(3).sum()
        t_tensor.pow(3).sum()
        torch.mul(tensor, random.randint(2, 10))
        torch.mul(t_tensor, random.randint(2, 10))
        torch.mul(tensor, tensor)
        torch.mul(t_tensor, t_tensor)

    # Resize to a thumbnail
    img.thumbnail(THUMB_SIZE)
    after_shape = img.size
    if verbose:
        print(f"augmented: shape:{img.size}| image tensor shape:{tensor.size()} transpose shape:{t_tensor.size()}")

    return None


# Define a Ray task to transform, augment and do some compute intensive tasks on an image
@ray.remote(num_cpus=15)
def augment_image_distributed(working_dir, complexity_score, fetch_image,s3=False, bucket_name="", object_key=""):
    Image.MAX_IMAGE_PIXELS = None
    img = Image.open(working_dir)
    return transform_image(img, fetch_image=fetch_image)

@ray.remote(num_cpus=15)
def augment_image_distributed_manual(image, complexity_score, fetch_image, bucket_name="", object_key=""):
    Image.MAX_IMAGE_PIXELS = None
    if not os.path.exists(image):
        # remote = True
        # time.sleep(complexity_score / 100000)
        # os.system("rsync --mkpath -a ubuntu@172.31.40.126:%s %s" %(image, image))

        if bucket_name!="" and object_key!="":
            node_type=os.getenv('LOCAL_NODE_TYPE')
            download_s3_folder(bucket_name,object_key,image,node_type)
        else:
            os.system(f"rsync -e 'ssh -o StrictHostKeyChecking=no' --mkpath -a {NODE_USER_NAME}@{DATA_IP}:{image} {image}")
    img = Image.open(image)
    return transform_image(img, fetch_image=fetch_image)

# def run_distributed(img_list_refs:List[object]) ->  List[Tuple[int, float]]:
#     return ray.get([augment_image_distributed.remote(img, False, working_dir=img) for img in img_list_refs])

# Check if dir exists. If so ignore download.
# Just assume we have done from a prior run
# if not os.path.exists(DATA_DIR):
#     os.mkdir(DATA_DIR)
#     print(f"downloading images ...")
#     for url in tqdm.tqdm(t_utils.URLS):
#         t_utils.download_images(url, DATA_DIR)
        
# Place all images into the object store. Since Ray tasks may be disributed 
# across machines, the DATA_DIR may not be present on a worker. However,
#placing them into the Ray distributed objector provides access to any 
# remote task scheduled on Ray worker
    
# images_list_refs = [t_utils.insert_into_object_store(image) for image in image_list]
# images_list_refs[:2]

# Iterate over batches, launching Ray task for each image within the processing
# batch
# for idx in BATCHES:
#     image_batch_list_refs = image_list[:idx]

print(f"\nRunning {len(image_list)} tasks distributed....")

# Run each one serially
start = time.perf_counter()

obj_refs = []
for img in image_list:
    img_name=img.split("/")[-1]
    
    cur_complexity = os.stat(img).st_size
    if use_scheduler:
        if use_s3:
            s3_object_path=s3_object_name+"/"+img_name
            working_dir_path= os.getcwd() + "/task_images_s3/"+img_name
            cur_complexity=get_s3_size(s3_bucket_name,s3_object_path)
            # print(s3_object_path," ",working_dir_path," ",cur_complexity)
            
            obj_refs.append(augment_image_distributed.remote(
            working_dir=working_dir_path, # TODO: Has to explicitly do so
            complexity_score=cur_complexity, 
            fetch_image=False,
            s3=use_s3,
            bucket_name=s3_bucket_name,
            object_key=s3_object_path
            
            ))
        else:
            
            obj_refs.append(augment_image_distributed.remote(
            working_dir=img, # TODO: Has to explicitly do so
            complexity_score=cur_complexity, 
            fetch_image=False
            ))
    else:
        if use_s3==True:
            s3_object_path=s3_object_name+"/"+img_name
            working_dir_path= os.getcwd() + "/task_images_s3/"+img_name
            
            cur_complexity=get_s3_size(s3_bucket_name,s3_object_path)
            obj_refs.append(augment_image_distributed_manual.remote(
            working_dir_path,
            cur_complexity, 
            False,
            bucket_name=s3_bucket_name,
            object_key=s3_object_path
            ))
        else:   
            obj_refs.append(augment_image_distributed_manual.remote(
            img,
            cur_complexity, 
            False
            ))
print(use_s3,use_scheduler)
distributed_results = ray.get(obj_refs)

end = time.perf_counter()
elapsed = end - start

print(elapsed)

# Keep track of batchs, execution times as a Tuple
# DISTRIBUTED_BATCH_TIMES.append((idx, round(elapsed, 2)))
# print(f"Distributed transformations/computations of {len(image_batch_list_refs)} images: {elapsed:.2f} sec")
