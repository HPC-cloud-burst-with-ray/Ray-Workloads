import json
import shutil
import os
from collections import defaultdict
# Specify the path to the JSON file
from const import *

json_file_path = DATASET_SOURCE_BASE + '/annotations/instances_train2017.json'



# Load the JSON file
with open(json_file_path, 'r') as f:
    data = json.load(f)

# Extract annotations and images
dic = defaultdict(list)
dic_images={}
annotations = data['annotations']
images=data['images']
info=data["info"]
licenses=data["licenses"]
categories=data['categories']
for i in annotations:
    dic[i['image_id']].append(i)
all_images=list(dic.keys())
for j in images:
    dic_images[j['id']]=j

# Sort annotations based on the "id" field
# sorted_annotations = sorted(annotations, key=lambda x: x['id'])
# print(sorted_annotations[0],sorted_annotations[1],sorted_annotations[-1])
def get_directory(dir_n,file_n):
    
    
    if os.path.exists(DATASET_DEST_BASE  + "/dataset_batch/"):
        os.system("rm -rf " + DATASET_DEST_BASE  + "/dataset_batch/")

    os.mkdir(DATASET_DEST_BASE  + "/dataset_batch/")
    
    for i in range(1,dir_n+1):
        data_new={}
        os.mkdir(DATASET_DEST_BASE + "/dataset_batch/"+str(i))
        dir_num=DATASET_DEST_BASE  + "/dataset_batch/"+str(i)+'/'
        new_annotations=[]
        new_images=[]
        for j in range(1,file_n+1):
            index=(i-1)*file_n+(j-1)
            
            image_id=all_images[index]
            file_name=(12-len(str(image_id)))*'0'+str(image_id)+".jpg"
            source_file=DATASET_SOURCE_BASE + "/train2017/"+file_name
            destination_file=dir_num+file_name
            shutil.copyfile(source_file, destination_file)
            new_annotations.extend(dic[image_id])
            new_images.append(dic_images[image_id])
        json_file_path=dir_num+"annotations.json"
        with open(json_file_path, 'w') as f:
            data_new ["licenses"]=licenses
            data_new["info"]=info
            data_new ["images"]=new_images
            data_new ["annotations"]=new_annotations 
            data_new ["categories"]=categories 
            
            json.dump(data_new, f)
        # generate sub annotations, each sub annotation contains a subset of images
        for k in range(NUM_BATCHES_IN_DIR):
            # each sub annotation contains a subset of images
            start = k * BATCH_SIZE
            end = (k + 1) * BATCH_SIZE
            sub_images = new_images[start:end]
            sub_annotations = []
            for sub_image in sub_images:
                sub_image_id = sub_image['id']
                sub_annotations.extend(dic[sub_image_id])
            sub_data = {
                "licenses": licenses,
                "info": info,
                "images": sub_images,
                "annotations": sub_annotations,
                "categories": categories
            }
            sub_json_file_path = dir_num + f"annotations_{k}.json"
            with open(sub_json_file_path, 'w') as f:
                json.dump(sub_data, f)
        print(f"Generated {NUM_BATCHES_IN_DIR} sub annotations in {dir_num}")


# (Num of directories, Num of files in each directory)
get_directory(NUM_DIR, NUM_IMG_IN_DIR)
            
