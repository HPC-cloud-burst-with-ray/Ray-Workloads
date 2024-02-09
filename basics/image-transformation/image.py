import os
import time
import logging
import math
import random
from PIL import Image
from pathlib import Path
from typing import Tuple, List

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import tqdm
import ray
import tasks_helper_utils as t_utils

if ray.is_initialized:
    ray.shutdown()
ray.init(logging_level=logging.ERROR)

DATA_DIR = Path(os.getcwd() + "/task_images")
BATCHES = [10, 20, 30, 40, 50]
SERIAL_BATCH_TIMES = []
DISTRIBUTED_BATCH_TIMES = []
# Define a Ray task to transform, augment and do some compute intensive tasks on an image
@ray.remote
def augment_image_distributed(image_ref: object, fetch_image,working_dir) -> List[object]:
    img_ref = Image.open(image_ref)
    return t_utils.transform_image(img_ref, fetch_image=fetch_image)

# Define function to run these transformation tasks distributed
# def run_distributed(img_list_refs:List[object]) ->  List[Tuple[int, float]]:
#     return ray.get([augment_image_distributed.remote(img, False,work_dir=DATA_DIR) for img in tqdm.tqdm(img_list_refs)])
def run_distributed(img_list_refs:List[object]) ->  List[Tuple[int, float]]:
    return ray.get([augment_image_distributed.remote(img, False,working_dir=img) for img in img_list_refs])

# Check if dir exists. If so ignore download.
# Just assume we have done from a prior run
# if not os.path.exists(DATA_DIR):
#     os.mkdir(DATA_DIR)
#     print(f"downloading images ...")
#     for url in tqdm.tqdm(t_utils.URLS):
#         t_utils.download_images(url, DATA_DIR)
        

# Fetch the the entire image list
image_list = list(DATA_DIR.glob("*.jpg"))
image_list[:2]


# Place all images into the object store. Since Ray tasks may be disributed 
# across machines, the DATA_DIR may not be present on a worker. However,
#placing them into the Ray distributed objector provides access to any 
# remote task scheduled on Ray worker
    
# images_list_refs = [t_utils.insert_into_object_store(image) for image in image_list]
# images_list_refs[:2]

# Iterate over batches, launching Ray task for each image within the processing
# batch
for idx in BATCHES:
    image_batch_list_refs = image_list[:idx]
    print(f"\nRunning {len(image_batch_list_refs)} tasks distributed....")
    
    # Run each one serially
    start = time.perf_counter()
    distributed_results = run_distributed(image_batch_list_refs)
    end = time.perf_counter()
    elapsed = end - start
    
    # Keep track of batchs, execution times as a Tuple
    DISTRIBUTED_BATCH_TIMES.append((idx, round(elapsed, 2)))
    print(f"Distributed transformations/computations of {len(image_batch_list_refs)} images: {elapsed:.2f} sec")
