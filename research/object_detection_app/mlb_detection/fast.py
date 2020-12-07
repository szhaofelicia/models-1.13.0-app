import io
import json
import math
import os

import numpy as np
import pandas as pd
from tqdm import tqdm
import csv

# activity='swing'
# numpy_dir='/media/felicia/Data/object_detection/data/%s/'%activity
# numpy_name='bbox_bbgame_swing'
# shard_id=0
# len_num_shards=6
# numpy_path = os.path.join(numpy_dir,'%s-%s-of-%s.npy' % (numpy_name,str(shard_id),str(len_num_shards)))
# numpy_dict= np.load(numpy_path,allow_pickle=True)
# numpy_dict=numpy_dict.item()


# images=numpy_dict['images']
# labels=numpy_dict['activity']
# steps=numpy_dict['steps']
# vd_names= numpy_dict['videos']

"""
ipython nbconvert --to script object_detection_mlb.ipynb
"""


"""
Statistic of bbox before and after filtering
"""


activity='swing'
num_shards=0
len_num_shards=6
NORMALIZED_FLOW=True
if NORMALIZED_FLOW:
    flow_thresh=[0.05,0.06,0.07] #[0.01,0.03,0.05,0.07,0.1,0.13,0.15,0.17,0.2] # [0.01,0.05,0.1,0.15,0.2]
else:
    flow_thresh=[1,3,5,7,10]


outputfolder='/media/felicia/Data/object_detection/statistic/%s'%activity
if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)


bbox_dir='/media/felicia/Data/object_detection/bbox/%s'%activity
bbox_name='bbox_bbgame_swing_person'
valid_name='valid_bbgame_swing'


FILTERING_TYPE= 'AVERAGE' # 'AVERAGE','ORIGINAL','WINDOW'
if FILTERING_TYPE=='AVERAGE':
    flow_name='ave_flow_bbgame_swing'
    flow_type='aveflow'
elif FILTERING_TYPE=='ORIGINAL':
    flow_name='wind_flow_bbgame_swing'
    flow_type='oriflow'
else:
    flow_name='wnd_flow_bbgame_swing'
    flow_type='wndflow'

AREA_CONDITION=True
if AREA_CONDITION:
    max_area=0.25 # default: 1
    area_name='area%s' % ("{:.0%}".format(max_area))
else:
    area_name=''
    


bbox_path = os.path.join(bbox_dir,'%s-%s-of-%s.npy' % (bbox_name,str(num_shards),str(len_num_shards)))
bbox_dict= np.load(bbox_path,allow_pickle=True)
bbox_dict=bbox_dict.item()

all_bboxes=bbox_dict['boxes']
all_scores=bbox_dict['scores']
all_classes=bbox_dict['classes']

image_dir='/media/felicia/Data/object_detection/data/%s'%activity
image_name='image_bbgame_swing'

image_path = os.path.join(image_dir,'%s-%s-of-%s.npy' % (image_name,str(num_shards),str(len_num_shards)))
print('Loading %s', image_path)
image_dict= np.load(image_path,allow_pickle=True)
image_dict=image_dict.item()

images=image_dict['images']
labels=image_dict['activity']
steps=image_dict['steps']
vd_names= image_dict['videos']


nframes=len(all_bboxes)
csv_keys=['id','video','step','activity','count_bf','count_af','score_bf','score_af']

for f in tqdm(flow_thresh):
    if NORMALIZED_FLOW:
        valid_path = os.path.join(bbox_dir,'%s-%s-of-%s_%s%s%s.npy' % (valid_name,str(num_shards),str(len_num_shards),flow_type,"{:.0%}".format(f),area_name))
    else:
        valid_path = os.path.join(bbox_dir,'%s-%s-of-%s_%s%s%s.npy' % (valid_name,str(num_shards),str(len_num_shards),flow_type,"{:02d}".format(f),area_name))
    valid_bbox_dict= np.load(valid_path,allow_pickle=True)
    valid_bbox_dict=valid_bbox_dict.item()
    
    valid_bboxes=valid_bbox_dict['boxes']
    valid_scores=valid_bbox_dict['scores']
    valid_classes=valid_bbox_dict['classes']

    csv_data=[]
    for i in range(nframes):
        temp_dict={
            'id':i,
            'video':vd_names[i],
            'step':steps[i],
            'activity':labels[i],
            'count_bf':len(all_scores[i]),
            'count_af':len(valid_scores[i]),
            'score_bf':all_scores[i],
            'score_af':valid_scores[i]
        }
        csv_data.append(temp_dict)
    
    if NORMALIZED_FLOW:
        csv_path = os.path.join(
            outputfolder,
            '%s-%s-of-%s_%s%s%s.csv' % (valid_name,str(num_shards),str(len_num_shards),flow_type,"{:.0%}".format(f),area_name))
    else:
        csv_path = os.path.join(
            outputfolder,
            '%s-%s-of-%s_%s%s%s.csv' % (valid_name,str(num_shards),str(len_num_shards),flow_type,"{:02d}".format(f),area_name))
    with open(csv_path,'w') as csvfile:
        writer=csv.DictWriter(csvfile,fieldnames=csv_keys)
        writer.writeheader()
        for data in csv_data:
            writer.writerow(data)


activity='swing'
record_time=  '11-09-20-44' # '11-07-01-17','11-09-19-13','11-09-20-44'
patch_dir='/media/felicia/Data/object_detection/patch/%s'%activity
patch_path = os.path.join(patch_dir,'%s.npy' % record_time)
patch_dict= np.load(patch_path,allow_pickle=True)
patch_dict=patch_dict.item()

vd_names=patch_dict['video']
steps=patch_dict['frame']
sorted_bbox=patch_dict['sorted_bbox']
crop_bbox=patch_dict['crop_bbox']
crop_patch=patch_dict['crop_patch']

outputfolder='/media/felicia/Data/object_detection/statistic/%s'%activity
csv_path = os.path.join(
    outputfolder,
    'patch_%s.csv' % record_time)

nframes=len(sorted_bbox)

csv_data=[]
for i in range(nframes):
    temp_dict={
        'id':i,
        'video':vd_names[i],
        'frame':steps[i],
        'sorted_bbox':sorted_bbox[i], # np.array: B*1
        'crop_bbox':crop_bbox[i], # np.array:B*1
        'crop_patch':crop_patch[i]
    }
    csv_data.append(temp_dict)
    
csv_keys=['id','video','frame','sorted_bbox','crop_bbox','crop_patch']

with open(csv_path,'w') as csvfile:
    writer=csv.DictWriter(csvfile,fieldnames=csv_keys)
    writer.writeheader()
    for data in csv_data:
        writer.writerow(data)




"""
## Load numpy file


activity='swing'
outputfolder='/media/felicia/Data/object_detection/optical_flow/%s'%activity

len_num_shards=6
i=0

output_filename = os.path.join(
    outputfolder,
    '%s-%s-of-%s.npy' % ('ori_flow_bbgame_swing',str(i),str(len_num_shards)))


numpy_dict= np.load(output_filename,allow_pickle=True)
numpy_dict=numpy_dict.item()

dense_flow=numpy_dict['dense_flow']
ave_score=numpy_dict['ave_score']



output_filename_ = os.path.join(
    outputfolder,
    '%s-%s-of-%s.npy' % ('wnd_05_flow_bbgame_swing',str(i),str(len_num_shards)))

numpy_dict= np.load(output_filename_,allow_pickle=True)
numpy_dict=numpy_dict.item()

wnd_flow=numpy_dict['wnd_flow']
ave_score_=numpy_dict['ave_score']


nframes=len(dense_flow)

for j in range(nframes):
    print(dense_flow[j][500][800])



for j in range(nframes):
    print(j,wnd_flow[j][500][800])

"""

"""
import yaml

config={
    'HEIGHT':{
        'WIDTH': 1280,
        'HEIGHT': 720
    },
    'PATCH':{
        'patchH': 350,
        'patchH': 450,
        'ADDAVE': True,
        'RMVAVE': True
    },
    'BBOX':{
        'AREA_CONDITION': True,
        'MAX_AREA': 0.25,
        'MIN_AREA': 0.01,
        'MIN_OCCUR': 3,
        'MIN_IOU': 0.1
    },
    'OPTICAL_FLOW':{
        'FLOW_TYPE': 'QUANTILE',
        'MIN_FLOW': 0.80    
    }
}

with open('./config_test.yml', 'w') as file:
    documents = yaml.dump(config, file)

"""

# """
## patch list to patch dictionary

activity='swing'
record_time=  '11-10-20-48' # min_iou=0.11
patch_dir='/media/felicia/Data/object_detection/patch/%s'%activity
t=0
len_num_shards=6

new_dict={}

for t in range(len_num_shards):

    patch_path = os.path.join(patch_dir,'%s-%s-of-%s_%s.npy' % ('patch_all',str(t),str(len_num_shards),record_time))
    patch_dict= np.load(patch_path,allow_pickle=True)
    patch_dict=patch_dict.item()

    vd_names=patch_dict['video']
    steps=patch_dict['frame']
    # sorted_bbox=patch_dict['sorted_bbox']
    # crop_bbox=patch_dict['crop_bbox']
    crop_patch=patch_dict['crop_patch']

    nframes=len(crop_patch)

    for i in range(nframes):
        key='%s_%s'%(vd_names[i].decode("utf-8"),"{:02d}".format(steps[i]))
        new_dict[key]={
            'left':crop_patch[i][0],
            'right':crop_patch[i][1]
        }
    
    # with open(os.path.join(patch_dir,'%s-%s-of-%s_%s.npy' % ('patch_dict',str(t),str(len_num_shards),record_time)),'wb') as file:
    #     np.save(file,new_dict)


with open(os.path.join(patch_dir,'%s_%s.npy' % ('patch_dict',record_time)),'wb') as file:
    np.save(file,new_dict)
    




# """