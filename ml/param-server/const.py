import os

NUM_DIR = 50
NUM_IMG_IN_DIR = 100


NODE_USER_NAME = "ec2-user"
DATA_IP= "10.0.0.132"
DATA_DIR = os.getcwd() + "/dataset_batch/*/"

DATASET_SOURCE_BASE = "/home/ec2-user"
DATASET_DEST_BASE = "/home/ec2-user/share/Ray-Workloads/ml/param-server"
