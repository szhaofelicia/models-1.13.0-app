import numpy as np
import os
import sys

import tensorflow.compat.v2 as tf
from distutils.version import StrictVersion
from collections import defaultdict
from io import StringIO
from PIL import Image

from absl import logging
from absl import app
from absl import flags

# This is needed since the notebook is stored in the object_detection folder.
sys.path.append("..")
from object_detection.utils import ops as utils_ops

if StrictVersion(tf.__version__) < StrictVersion('1.9.0'):
  raise ImportError('Please upgrade your TensorFlow installation to v1.9.* or later!')

from utils import label_map_util
from utils import visualization_utils as vis_util

from tqdm import tqdm
import copy


flags.DEFINE_string('model_name', 'ssd_inception_v2_coco_2018_01_28', 'The name of pre-trained model.')
flags.DEFINE_string('data_label', 'mscoco_label_map.pbtxt', 'List of correct labels for each box.')
flags.DEFINE_string('output_dir', '/media/felicia/Data/object_detection/bbox/%s', 'The path to output_folder.')
flags.DEFINE_string('acivity', 'swing', 'The class of test images.')
flags.DEFINE_integer('batch', 20, 'The size of batch.')
flags.DEFINE_string('name', 'bbox_bbgame_swing_person', 'Name of the dataset being created. This will'
                    'be used as a prefix.') # bbox_bbgame_swing

FLAGS = flags.FLAGS

gfile = tf.io.gfile

WIDTH=1280
HEIGHT=720
max_boxes=20 # default: 10
min_score=.1

def run_inference_for_minibatch_images(images, graph):
  """
  images: numpy.array- B*H *W *C
  """
  with graph.as_default():
    with tf.compat.v1.Session() as sess:
      ops = tf.compat.v1.get_default_graph().get_operations()
      all_tensor_names = {output.name for op in ops for output in op.outputs}
      tensor_dict = {}
      for key in [
        'num_detections', 'detection_boxes', 'detection_scores',
        'detection_classes', 'detection_masks'
      ]:
        tensor_name = key + ':0'
        if tensor_name in all_tensor_names:
            tensor_dict[key] = tf.compat.v1.get_default_graph().get_tensor_by_name(tensor_name)
      image_tensor = tf.compat.v1.get_default_graph().get_tensor_by_name('image_tensor:0')

      # Run inference
      output_dict = sess.run(tensor_dict,
                            feed_dict={image_tensor: images})
  return output_dict

def valid_boxes_on_minibatch_images(boxes,classes,scores,max_boxes_to_draw,min_score_thresh,selected_class=None):
    valid_output={}
    b=boxes.shape[0]

    minibatch_boxes=[]
    minibatch_scores=[]
    minibatch_classes=[]
    for i in range(b):
      single_boxes=[]
      single_scores=[]
      single_classes=[]
      for j in range(min(max_boxes_to_draw,boxes.shape[0])):
        if scores[i][j]> min_score_thresh and (not selected_class or classes[i][j]==selected_class):
            ymin, xmin, ymax, xmax=boxes[i][j]
            (left, right, top, bottom) = (xmin * WIDTH, xmax * WIDTH, ymin * HEIGHT, ymax * HEIGHT)
            single_boxes.append([left, right, top, bottom])
            single_scores.append(scores[i][j])
            single_classes.append(classes[i][j])
      minibatch_boxes.append(single_boxes)
      minibatch_scores.append(single_scores)
      minibatch_classes.append(single_classes)
    return minibatch_boxes, minibatch_scores,minibatch_classes
            

def main(_):
 # What model to download.
  MODEL_NAME = FLAGS.model_name

  # Path to frozen detection graph. This is the actual model that is used for the object detection.
  PATH_TO_FROZEN_GRAPH = 'pretrained/'+ MODEL_NAME + '/frozen_inference_graph.pb'

  # List of the strings that is used to add correct label for each box.
  PATH_TO_LABELS = os.path.join('data', FLAGS.data_label)

  ACTIVITY=FLAGS.acivity
  # PATH_TO_TEST_IMAGES_DIR = FLAGS.path_to_test_images_dir%ACTIVITY

  BATCH=FLAGS.batch
  outputfolder=FLAGS.output_dir%ACTIVITY

  if not os.path.exists(outputfolder):
      os.makedirs(outputfolder)

  detection_graph = tf.Graph()
  with detection_graph.as_default():
    od_graph_def = tf.compat.v1.GraphDef()
    with gfile.GFile(PATH_TO_FROZEN_GRAPH, 'rb') as fid:
      serialized_graph = fid.read()
      od_graph_def.ParseFromString(serialized_graph)
      tf.import_graph_def(od_graph_def, name='')

  category_index = label_map_util.create_category_index_from_labelmap(PATH_TO_LABELS, use_display_name=True)

  activity='swing'
  numpy_dir='/media/felicia/Data/object_detection/data/%s/'%activity
  numpy_name='image_bbgame_swing'
  len_num_shards=6

  # Size, in inches, of the output images.

  for i in range(len_num_shards):
    valid_boxes=[]
    valid_scores=[]
    valid_classes=[]

    numpy_path = os.path.join(numpy_dir,'%s-%s-of-%s.npy' % (numpy_name,str(i),str(len_num_shards)))

    logging.info('Loading %s', numpy_path)

    numpy_dict= np.load(numpy_path,allow_pickle=True)
    numpy_dict=numpy_dict.item()

    images=numpy_dict['images']
    labels=numpy_dict['activity']
    steps=numpy_dict['steps']
    vd_names= numpy_dict['videos']

    nbatch=len(images)//BATCH+(1 if len(images)%BATCH>0 else 0)

    for j in tqdm(range(nbatch)):
      idx_st=j*BATCH 
      idx_ed=(j+1)*BATCH if (j+1)*BATCH < len(images) else len(images)
      image_batch=images[idx_st:idx_ed] # B*H*W*C

      output_dict = run_inference_for_minibatch_images(image_batch, detection_graph)

      minibatch_boxes, minibatch_scores,minibatch_classes=valid_boxes_on_minibatch_images( 
          output_dict['detection_boxes'],
          output_dict['detection_classes'],
          output_dict['detection_scores'],
          max_boxes,
          min_score,
          selected_class=1
          )
      
      valid_boxes+=minibatch_boxes
      valid_scores+=minibatch_scores
      valid_classes+=minibatch_classes

    valid_output={
      'boxes':valid_boxes,
      'scores':valid_scores,
      'classes':valid_classes
    }

    output_filename = os.path.join(
      outputfolder,
      '%s-%s-of-%s.npy' % (FLAGS.name,str(i),str(len_num_shards)))
    
    with open(output_filename,'wb') as file:
      np.save(file,valid_output)


if __name__ == '__main__':
  app.run(main)

