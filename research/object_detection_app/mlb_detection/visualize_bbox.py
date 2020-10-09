import numpy as np
import os
import sys
import tensorflow as tf

import matplotlib.pyplot as plt
import matplotlib as mpl


















if not os.path.exists(outputfolder):
    os.makedirs(outputfolder)

for j in range(image_batch.shape[0]):
    
    vis_util.visualize_ordered_boxes_and_labels_on_image_array(
        image_temp0[j],
        valid_output1['boxes'][j], # 4 * 4, ymin, xmin, ymax, xmax = box
        box_classes[j], # (4,)
        valid_output1['scores'][j], # (4,)
        category_index,
        use_normalized_coordinates=True,
        max_boxes_to_draw=max_boxes,
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
    # plt.figure(figsize=IMAGE_SIZE)
    fig,ax=plt.subplots(2,2,figsize=IMAGE_SIZE)
    ax[0,0].imshow(image_temp0[j])
    ax[0,1].imshow(image_temp1[j])

    # plt.suptitle('min_area={0:.3f} max_area={1:.3f}'.format(min_area,max_area))

    #display rescaled depth
    rescaled = (image_depth[j] - np.min(image_depth[j]))/(np.max(image_depth[j])-np.min(image_depth[j]))
    depth=ax[1,0].imshow(plasma(rescaled)[:,:,:3],cmap='plasma')
    plt.colorbar(depth, ticks=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1.0],orientation='vertical')
    ave=np.mean(rescaled,axis=(0,1))
    med=np.median(rescaled,axis=(0,1))
    ax[1,1].hist(rescaled.ravel(),100,[0,1])
    ax[1,1].set_title('ave='+str(ave)+' median='+str(med))

    # ax[1,0].imshow(image_foreground[i],cmap='gray', vmin = 0, vmax = 255) # B * H * W * 3 
    # ax[1,1].imshow(gray_foreground[i],cmap='gray', vmin = 0, vmax = 255) # B * H * W 

    threshold=np.quantile(rescaled,q=q,axis=(0,1))
    plt.suptitle('{0:.3f} qunatile={1:.4f}'.format(q,threshold))
    
    plt.savefig(outputfolder+filenames[i*BATCH+j]+'.png',doi=100)
    plt.close()
    
    # plt.show(block=False)
    # (left, right, top, bottom) = (xmin * WIDTH, xmax * WIDTH, ymin * HEIGHT, ymax * HEIGHT)
    # if q==0.6:
    print(i*BATCH+j,'Area',normalized_area[j])
    print(i*BATCH+j,'Accuracy', valid_output0['scores'][j])