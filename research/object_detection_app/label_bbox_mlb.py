import numpy as np
import os
import six.moves.urllib as urllib
import sys
import tarfile
import tensorflow as tf
import zipfile

from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
import matplotlib.pyplot as plt
import matplotlib as mpl
from PIL import Image

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')


from utils import label_map_util
from utils import visualization_utils as vis_util

from tqdm import tqdm
import time
import cv2

import copy
import pickle
import json

# What model to download.
MODEL_NAME = 'ssd_inception_v2_coco_2018_01_28'

# Path to frozen detection graph. This is the actual model that is used for the object detection.
PATH_TO_FROZEN_GRAPH = 'pretrained/'+ MODEL_NAME + '/frozen_inference_graph.pb'

# List of the strings that is used to add correct label for each box.
PATH_TO_LABELS = os.path.join('data', 'mscoco_label_map.pbtxt')

detection_graph = tf.Graph()
with detection_graph.as_default():
    od_graph_def = tf.GraphDef()
    with tf.gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
        serialized_graph = fid.read()
        od_graph_def.ParseFromString(serialized_graph)
        tf.import_graph_def(od_graph_def, name='')

category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

HEIGHT=720
WIDTH=1280


image_row=np.array([[r for c in range(WIDTH)] for r in range(HEIGHT)]) # H*W
image_col=np.array([[c for c in range(WIDTH)] for r in range(HEIGHT)])

PATH_TO_TEST_IMAGES_DIR = '/media/felicia/Data/mlb-youtube/frames_continuous/swing'
PICKLE_DIR='/media/felicia/Data/baseballplayers/pickles/'


videonames=pickle.load(open(PICKLE_DIR+'swing_videos.pkl','rb'))



def run_inference_for_minibatch_images(images, graph):
    """
    images: numpy.array- B*H *W *C
    """
    with graph.as_default():
        with tf.Session() as sess:
            ops = tf.get_default_graph().get_operations()
            all_tensor_names = {output.name for op in ops for output in op.outputs}
            tensor_dict = {}
      
            for key in [
                'num_detections', 'detection_boxes', 'detection_scores',
                'detection_classes', 'detection_masks'
            ]:
                tensor_name = key + ':0'
                if tensor_name in all_tensor_names:
                    tensor_dict[key] = tf.get_default_graph().get_tensor_by_name(tensor_name)

            image_tensor = tf.get_default_graph().get_tensor_by_name('image_tensor:0')
            # Run inference
            output_dict = sess.run(tensor_dict,
                                    feed_dict={image_tensor: images})
    return output_dict



def valid_boxes_on_minibatch_images(boxes,classes,scores,max_boxes_to_draw,min_score_thresh,min_area,max_area):
    """
    boxes: B*100 *4
    classes: B*100
    scores: B*100 --> B*4
    """
    valid_output={}
    valid_boxes=[]
    valid_scores=[]

    b=boxes.shape[0]
    class_mask=(classes==1).astype(np.int)
    score_mask=(scores>min_score_thresh).astype(np.int)

    normalized_h=boxes[:,:,2]-boxes[:,:,0]  #  B*4:ymax-ymin
    normalized_w=boxes[:,:,3]-boxes[:,:,1]  #  B*4:xmax-xmin

    normalized_hc=0.5*boxes[:,:,2]+0.5*boxes[:,:,0]  #  B*4:(ymax+ymin)/2
    normalized_wc=0.5*boxes[:,:,3]+0.5*boxes[:,:,1]  #  B*4:(xmax+xmin)/2

    normalized_area=np.multiply(normalized_w,normalized_h)
    minarea_mask=(normalized_area> min_area).astype(np.int) # plus | center_mask
    maxarea_mask=(normalized_area< max_area).astype(np.int)

    center_mask = ((normalized_hc > (240/HEIGHT) ) & ( normalized_hc < (460/HEIGHT) )).astype(np.int) 
    center_mask = center_mask & ((normalized_wc > (400/WIDTH) ) & ( normalized_wc < (840/WIDTH) )).astype(np.int) 

    area_mask = ( minarea_mask | center_mask) & maxarea_mask

    valid_mask = class_mask & score_mask & area_mask  # B *100

  
    for i in range(b):
        valid_box=boxes[i][valid_mask[i]==1][:max_boxes_to_draw,:]
        valid_score=scores[i][valid_mask[i]==1][:max_boxes_to_draw]
        if valid_box.shape[0]<max_boxes_to_draw:
            valid_box=np.pad(valid_box,((0,max_boxes_to_draw-valid_box.shape[0]),(0,0)),'constant',constant_values=((0,0),(0,0)))
            valid_score=np.pad(valid_score,(0,max_boxes_to_draw-valid_score.shape[0]),'constant',constant_values=(0,0))
        valid_boxes.append(valid_box)
        valid_scores.append(valid_score)

    valid_output['boxes']=np.array(valid_boxes)
    valid_output['scores']=np.array(valid_scores)

    return  valid_output



def foreground_mask_image_array(image_depth,boxes,scores,max_boxes_to_draw=5,alpha=0.2,quantile=0.5):
    valid_output={}
    valid_boxes=[]
    valid_scores=[]
    
    batchsize=boxes.shape[0]
    nboxes=boxes.shape[1]
    batch_row=np.array([[image_row for i in range(nboxes)] for j in range(batchsize)])
    batch_col=np.array([[image_col for i in range(nboxes)] for j in range(batchsize)])

    # (left, right, top, bottom) = (xmin * WIDTH, xmax * WIDTH, ymin * HEIGHT, ymax * HEIGHT)
    normalized_h=boxes[:,:,2]-boxes[:,:,0]  #  B*4:ymax-ymin
    normalized_w=boxes[:,:,3]-boxes[:,:,1]  #  B*4:xmax-xmin
    areas=np.multiply(normalized_w,normalized_h)

    normalized_hc=0.5*boxes[:,:,2]+0.5*boxes[:,:,0]  #  B*4:(ymax+ymin)/2
    normalized_wc=0.5*boxes[:,:,3]+0.5*boxes[:,:,1]  #  B*4:(xmax+xmin)/2

    center_mask = ((normalized_hc > (150/HEIGHT) ) & ( normalized_hc < (460/HEIGHT) )).astype(np.int) 
    center_mask = center_mask & ((normalized_wc > (400/WIDTH) ) & ( normalized_wc < (975/WIDTH) )).astype(np.int) 

    boxes_xmin=boxes[:,:,1] # B*nboxes
    boxes_xmin=np.tile(boxes_xmin[:,:,np.newaxis,np.newaxis],(1,1,HEIGHT,WIDTH))
    boxes_xmax=boxes[:,:,3]
    boxes_xmax=np.tile(boxes_xmax[:,:,np.newaxis,np.newaxis],(1,1,HEIGHT,WIDTH))
    boxes_ymin=boxes[:,:,0]
    boxes_ymin=np.tile(boxes_ymin[:,:,np.newaxis,np.newaxis],(1,1,HEIGHT,WIDTH))
    boxes_ymax=boxes[:,:,2]
    boxes_ymax=np.tile(boxes_ymax[:,:,np.newaxis,np.newaxis],(1,1,HEIGHT,WIDTH))

    row_mask=np.greater_equal(batch_row,HEIGHT*boxes_ymin) & np.less_equal(batch_row,HEIGHT*boxes_ymax)
    col_mask=np.greater_equal(batch_col,WIDTH*boxes_xmin) & np.less_equal(batch_col,WIDTH*boxes_xmax)
    box_mask=row_mask& col_mask # B*10 *H*W


    # unrescaled threshold
    threshold=np.quantile(image_depth,q=quantile,axis=(1,2)) # B*H*W
    threshold=np.repeat(threshold[:,np.newaxis],10,axis=1)  # B* 10

    image_depth=np.repeat(image_depth[:,np.newaxis,:,:],10,axis=1) # B*10*H*W
    depth_region=np.sum(np.multiply(image_depth,box_mask),axis=(2,3)) # B* 10
    depth_mask=(np.divide(depth_region,HEIGHT*WIDTH*areas)<=threshold).astype('uint8')  # B* 10

    depth_mask=depth_mask | center_mask
  
    for i in range(batchsize):
        valid_box=boxes[i][depth_mask[i]==1][:max_boxes_to_draw,:]
        valid_score=scores[i][depth_mask[i]==1][:max_boxes_to_draw]
        if valid_box.shape[0]<max_boxes_to_draw:
            valid_box=np.pad(valid_box,((0,max_boxes_to_draw-valid_box.shape[0]),(0,0)),'constant',constant_values=((0,0),(0,0)))
            valid_score=np.pad(valid_score,(0,max_boxes_to_draw-valid_score.shape[0]),'constant',constant_values=(0,0))
        valid_boxes.append(valid_box)
        valid_scores.append(valid_score)
    valid_output['boxes']=np.array(valid_boxes)
    valid_output['scores']=np.array(valid_scores)
    
    return valid_output


BATCH=25
max_boxes=10
min_score=0
min_area= 0.015 #0.01, 0.015, 0.02 
max_area= 0.19 #0.15
# ALL=1000
nvideos=10
nframes=19
q=0.640

# Size, in inches, of the output images.
IMAGE_SIZE = (24, 8) # (12, 8)
# plasma = plt.get_cmap('plasma')

# depth=predict(model, inputs)

# load depth estimation model

outputfolder='/media/felicia/Data/baseballplayers/swing/'



for i in tqdm(range(100,len(videonames))):
    v=videonames[i]
    print('====================================')
    print('\n'+v)

    bbox_folder=outputfolder+v+'/'

    if not os.path.exists(bbox_folder):
        os.makedirs(bbox_folder)
    # else:
    #     continue

    depth_dict=pickle.load(open(PICKLE_DIR+'depth_outputs_{:03d}.pkl'.format(i),'rb')) # i-th video
    filenames=depth_dict['filenames']
    depth_outputs=depth_dict['outputs'] # B * 240 * 320 *1 , unrescaled
    frame_depth=depth_dict['frame_depth']
    
    image_depth=list() 
    image_batch=list() 
    for j in range(nframes):
        image=Image.open(os.path.join(PATH_TO_TEST_IMAGES_DIR,filenames[j]+'.jpg'))
        image_np=np.asarray(image).reshape(HEIGHT, WIDTH,3).astype(np.uint8)
        image_batch.append(np.expand_dims(image_np,0))
        depth=cv2.resize(frame_depth[filenames[j]],dsize=(WIDTH,HEIGHT),  interpolation = cv2.INTER_AREA)
        image_depth.append(np.expand_dims(depth,axis=0))
    image_batch=np.concatenate(image_batch,axis=0) # B*H*W*C
    image_depth=np.concatenate(image_depth,axis=0) # B*H*W*C

    output_dict = run_inference_for_minibatch_images(image_batch, detection_graph)
    print('\nremove invalid boxes')
    valid_output=valid_boxes_on_minibatch_images( 
        output_dict['detection_boxes'],
        output_dict['detection_classes'],
        output_dict['detection_scores'],
        max_boxes,
        min_score,
        min_area,
        max_area)

    box_classes=np.ones((image_batch.shape[0],max_boxes)).astype(np.int)

    print('\nremove boxes on the background')
    foreground_output=foreground_mask_image_array(image_depth,valid_output['boxes'],valid_output['scores'],max_boxes_to_draw=10,quantile=q)

    normalized_boxes=foreground_output['boxes'] #B*4* 4: ymin, xmin, ymax, xmax 
    normalized_h=normalized_boxes[:,:,2]-normalized_boxes[:,:,0]  #  B*4:ymax-ymin
    normalized_w=normalized_boxes[:,:,3]-normalized_boxes[:,:,1]  #  B*4:xmax-xmin
    normalized_area=np.multiply(normalized_w,normalized_h)

    image_temp0=copy.deepcopy(image_batch)
    image_temp1=copy.deepcopy(image_batch)

    for j in range(nframes):
        vis_util.visualize_ordered_boxes_and_labels_on_image_array(
            image_temp0[j],
            output_dict['detection_boxes'][j], # 4 * 4, ymin, xmin, ymax, xmax = box
            output_dict['detection_classes'][j], # (4,)
            output_dict['detection_scores'][j], # (4,)
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=20,
            min_score_thresh=min_score,
            line_thickness=8)
        vis_util.visualize_ordered_boxes_and_labels_on_image_array(
            image_temp1[j],
            foreground_output['boxes'][j], # 4 * 4, ymin, xmin, ymax, xmax = box
            box_classes[j], # (4,)
            foreground_output['scores'][j], # (4,)
            category_index,
            use_normalized_coordinates=True,
            max_boxes_to_draw=max_boxes,
            min_score_thresh=min_score,
            line_thickness=8)
        fig,ax=plt.subplots(1,2,figsize=IMAGE_SIZE)
        ax[0].imshow(image_temp0[j])
        ax[1].imshow(image_temp1[j])

        #display rescaled depth
        # rescaled = (image_depth[j] - np.min(image_depth[j]))/(np.max(image_depth[j])-np.min(image_depth[j]))
        # depth=ax[1,0].imshow(plasma(rescaled)[:,:,:3],cmap='plasma')
        # plt.colorbar(depth, ticks=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],orientation='vertical')
        # ave=np.mean(rescaled,axis=(0,1))
        # med=np.median(rescaled,axis=(0,1))
        # ax[1,1].hist(rescaled.ravel(),100,[0,1])
        # ax[1,1].set_title('ave='+str(ave)+' median='+str(med))

        # threshold=np.quantile(rescaled,q=q,axis=(0,1))
        # plt.suptitle('{0:.3f} qunatile={1:.4f}'.format(q,threshold))

        plt.suptitle('min_area={0:.3f} max_area={1:.3f} quantile={2:.3f}'.format(min_area,max_area,q))
        
        plt.savefig(bbox_folder+filenames[j]+'.png',doi=100)
        plt.close()
        
        # print(i*BATCH+j,'Area',normalized_area[j])
        # print(i*BATCH+j,'Accuracy', foreground_output['scores'][j])

    bbox_json={}
    bbox_json[v]={}
    bbox_json[v]['bbox']=foreground_output['boxes'].tolist()
    bbox_json[v]['accuracy']=foreground_output['scores'].tolist()
    bbox_json[v]['area']=normalized_area.tolist()
    
    with open('/media/felicia/Data/baseballplayers/jsons/'+v+'.json','w') as file:
        json.dump(bbox_json,file)



"""
no valid box-- 0
*pitcher=1
^batter=2
*catcher=3
-umpire=4

    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],

    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],

    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],

    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0],
    [0,0,0,0,0,0,0,0,0,0]

"""

