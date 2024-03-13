import os

NUM_DIR = 50
NUM_IMG_IN_DIR = 50
MAX_BANDWIDTH_HPC=629145600
# sub batch in one dir
NUM_BATCHES_IN_DIR = 1

assert NUM_IMG_IN_DIR % NUM_BATCHES_IN_DIR == 0

BATCH_SIZE = NUM_IMG_IN_DIR // NUM_BATCHES_IN_DIR


NODE_USER_NAME = "ec2-user"
# DATA_IP= "10.0.0.54"
# DATA_IP is env variable HEAD_NODE_IP
DATA_IP = os.environ.get("HEAD_NODE_IP", None)
if DATA_IP is None:
    print("No IP found for data node. Exiting")
    sys.exit(1)

DATA_DIR_BASE = os.getcwd() + "/dataset_batch/"
DATA_DIR = os.getcwd() + "/dataset_batch/*/"

DATA_DIR_S3_BASE = os.getcwd() + "/dataset_batch_s3/"

DATASET_SOURCE_PATH = "/home/ec2-user/PASS_dataset"
DATASET_DEST_PATH = "/home/ec2-user/share/Ray-Workloads/ml/param-server-PASS/dataset_batch"
