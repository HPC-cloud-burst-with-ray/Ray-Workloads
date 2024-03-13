import glob
import random
import os
from const import *

INPUT_PATH = DATASET_SOURCE_PATH
OUTPUT_PATH = DATASET_DEST_PATH

if __name__ == "__main__":
    classes = glob.glob(INPUT_PATH + "/*")
    num_classes = len(classes)
    selected_image_count = [{i : 0 for i in classes} for _ in range(NUM_DIR)]

    # Generate random numbers to select class
    for i in range(NUM_DIR):
        for _ in range(NUM_IMG_IN_DIR):
            selected_class = random.randint(0, num_classes - 1)
            selected_image_count[i][classes[selected_class]] += 1

    # Create output directory
    if os.path.exists(OUTPUT_PATH):
        os.system("rm -rf " + OUTPUT_PATH)
    os.mkdir(OUTPUT_PATH)

    for i in range(NUM_DIR):
        os.mkdir(OUTPUT_PATH + "/" + str(i))
    
    # Copy image to each dir
    for i in range(num_classes):
        cur_class = classes[i]
        images = glob.glob(cur_class + "/*")
        offset = 0

        for j in range(NUM_DIR):
            cur_selected_count = selected_image_count[j][cur_class]
            if cur_selected_count != 0:
                cur_class_path = OUTPUT_PATH + "/" + str(j) + "/" + str(i)
                os.mkdir(cur_class_path)
                
                for k in range(offset, offset + cur_selected_count):
                    os.system("cp " + images[k] + " " + cur_class_path + "/")
                
                offset += cur_selected_count
