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
    flow_thresh=[0.01,0.03,0.05,0.07,0.1,0.13,0.15,0.17,0.2] # [0.01,0.05,0.1,0.15,0.2]
else:
    flow_thresh=[1,3,5,7,10]


outputfolder='/media/felicia/Data/object_detection/statistic/%s'%activity
if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)


bbox_dir='/media/felicia/Data/object_detection/bbox/%s'%activity
bbox_name='bbox_bbgame_swing_person'
valid_name='valid_bbgame_swing'

AVERAGE_FLOW=False
if AVERAGE_FLOW:
    flow_type='aveflow'
else:
    flow_type='oriflow'


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
        valid_path = os.path.join(bbox_dir,'%s-%s-of-%s_%s%s.npy' % (valid_name,str(num_shards),str(len_num_shards),flow_type,"{:.0%}".format(f)))
    else:
        valid_path = os.path.join(bbox_dir,'%s-%s-of-%s_%s%s.npy' % (valid_name,str(num_shards),str(len_num_shards),flow_type,"{:02d}".format(f)))
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
            '%s-%s-of-%s_%s%s.csv' % (valid_name,str(num_shards),str(len_num_shards),flow_type,"{:.0%}".format(f)))
    else:
        csv_path = os.path.join(
            outputfolder,
            '%s-%s-of-%s_%s%s.csv' % (valid_name,str(num_shards),str(len_num_shards),flow_type,"{:02d}".format(f)))
    with open(csv_path,'w') as csvfile:
        writer=csv.DictWriter(csvfile,fieldnames=csv_keys)
        writer.writeheader()
        for data in csv_data:
            writer.writerow(data)



