import os

NUM_DIR = 50
NUM_IMG_IN_DIR = 50
MAX_BANDWIDTH_HPC=200
# sub batch in one dir
NUM_BATCHES_IN_DIR = 1

assert NUM_IMG_IN_DIR % NUM_BATCHES_IN_DIR == 0

BATCH_SIZE = NUM_IMG_IN_DIR // NUM_BATCHES_IN_DIR


NODE_USER_NAME = "ec2-user"
DATA_IP= "10.0.0.132"
DATA_DIR_BASE = os.getcwd() + "/dataset_batch/"
DATA_DIR = os.getcwd() + "/dataset_batch/*/"

DATA_DIR_S3_BASE = os.getcwd() + "/dataset_batch_s3/"

DATASET_SOURCE_BASE = "/home/ec2-user"
DATASET_DEST_BASE = "/home/ec2-user/share/Ray-Workloads/ml/param-server"
