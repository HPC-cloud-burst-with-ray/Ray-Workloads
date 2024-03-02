import json
import shutil
import os
from collections import defaultdict
# Specify the path to the JSON file
json_file_path = '/home/ubuntu/train_ml/annotations/instances_train2017.json'

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
    os.mkdir("./dataset_batch/")
    for i in range(1,dir_n+1):
        data_new={}
        os.mkdir("./dataset_batch/"+str(i))
        dir_num="./dataset_batch/"+str(i)+'/'
        new_annotations=[]
        new_images=[]
        for j in range(1,file_n+1):
            index=(i-1)*file_n+(j-1)
            
            image_id=all_images[index]
            file_name=(12-len(str(image_id)))*'0'+str(image_id)+".jpg"
            source_file="./train2017/"+file_name
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


# (Num of directories, Num of files in each directory)
get_directory(5,16)
            
