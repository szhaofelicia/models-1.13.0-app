"""
input_directory='/media/felicia/Data/mlb-youtube/frames/'

"""

import numpy as np
import cv2
import os
from tqdm import tqdm
from PIL import Image


input_directory='/media/felicia/Data/mlb-youtube/frames/'
output_directory='/home/felicia/research/models-1.13.0/research/object_detection/mlb_dataset/dataset.record'

"""
# data_path = os.path.join(input_directory,'*g')
# files = glob.glob(data_path)
data = [] # #frames=1070
for f in tqdm(os.listdir(input_directory)):
    img = cv2.imread(input_directory+f)
    # print(img.shape)
    data.append(img)
    # break
"""
# (720, 1280, 3),height=720, width=1280,RGB


"""
Convert to TFrecord
"""


import tensorflow as tf

from utils import dataset_util


flags = tf.app.flags
flags.DEFINE_string('image_path', '',input_directory)
flags.DEFINE_string('output_path', '',output_directory)

FLAGS = flags.FLAGS

nframes=1070


def create_tf_example(f,inputpath): # inputpath+filename->example
    # TODO(user): Populate the following variables from your example.
    height = 720 # Image height
    width = 1280 # Image width
    filename = f.split('.')[0].encode('utf8') # Filename of the image. Empty if image is not from file
    image_format = b'jpg' # b'jpeg' or b'png'

    # encoded_image_data = None # Encoded image bytes
    with tf.gfile.GFile(os.path.join(inputpath, f), 'rb') as fid:
        encoded_image_data = fid.read()
    
    image=Image.open(inputpath+f)
    width,height=image.size 




    xmins = [0] # List of normalized left x coordinates in bounding box (1 per box)
    xmaxs = [0] # List of normalized right x coordinates in bounding box
                # (1 per box)
    ymins = [0] # List of normalized top y coordinates in bounding box (1 per box)
    ymaxs = [0] # List of normalized bottom y coordinates in bounding box
                # (1 per box)
    classes_text = ['Human'.encode('utf8')] # List of string class name of bounding box (1 per box)
    classes = [1] # List of integer class id of bounding box (1 per box)

    tf_example = tf.train.Example(features=tf.train.Features(feature={
        'image/height': dataset_util.int64_feature(height),
        'image/width': dataset_util.int64_feature(width),
        'image/filename': dataset_util.bytes_feature(filename),
        'image/source_id': dataset_util.bytes_feature(filename),
        'image/encoded': dataset_util.bytes_feature(encoded_image_data),
        'image/format': dataset_util.bytes_feature(image_format),
        'image/object/bbox/xmin': dataset_util.float_list_feature(xmins),
        'image/object/bbox/xmax': dataset_util.float_list_feature(xmaxs),
        'image/object/bbox/ymin': dataset_util.float_list_feature(ymins),
        'image/object/bbox/ymax': dataset_util.float_list_feature(ymaxs),
        'image/object/class/text': dataset_util.bytes_list_feature(classes_text),
        'image/object/class/label': dataset_util.int64_list_feature(classes),
    }))
    return tf_example


def main(_):
    writer = tf.python_io.TFRecordWriter(output_directory)

    # TODO(user): Write code to read in your dataset to examples variable


    for f in tqdm(os.listdir(input_directory)): # f:filename
        tf_example = create_tf_example(f,input_directory)
        writer.write(tf_example.SerializeToString())

    writer.close()


if __name__ == '__main__':
    tf.app.run()