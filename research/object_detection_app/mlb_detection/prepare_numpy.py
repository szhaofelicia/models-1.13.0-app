
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import io
import json
import math
import os

import numpy as np
from PIL import Image
from PIL import ImageFile
import tensorflow.compat.v2 as tf

from absl import logging
from absl import app
from absl import flags

import cv2




flags.DEFINE_string('acivity', 'swing', 'The class of test images.')
flags.DEFINE_string('input_dir', '/media/felicia/Data/mlb-youtube/%s_videos/rm_noise/videos', 'Path to videos.')
flags.DEFINE_string('name', 'image_bbgame_swing', 'Name of the dataset being created. This will'
                    'be used as a prefix.')
flags.DEFINE_string('file_pattern', '*.mp4', 'Pattern used to searh for files'
                    'in the given directory.')
flags.DEFINE_string('label_file', None, 'Provide a corresponding labels file'
                    'that stores per-frame or per-sequence labels. This info'
                    'will get stored.')
flags.DEFINE_string('output_dir', '/media/felicia/Data/object_detection/data/%s/', 'Output directory where'
                    'tfrecords will be stored.')
flags.DEFINE_integer('files_per_shard', 50, 'Number of videos to store in a'
                     'shard.')
flags.DEFINE_boolean('rotate', False, 'Rotate videos by 90 degrees before'
                     'creating tfrecords')
flags.DEFINE_boolean('resize', True, 'Resize videos to a given size.')
flags.DEFINE_integer('width', 1280, 'Width of frames in the TFRecord.')
flags.DEFINE_integer('height', 720, 'Height of frames in the TFRecord.')
flags.DEFINE_list(
    'frame_labels', '', 'Comma separated list of descriptions '
    'for labels given on a per frame basis. For example: '
    'winding_up,early_cocking,acclerating,follow_through')
flags.DEFINE_integer('action_label',0 , 'Action label of all videos.') # swing:0, ball:1, strike:2, foul:3, hit:4
flags.DEFINE_integer('expected_segments', -1, 'Expected number of segments.')
flags.DEFINE_integer('fps', 10, 'Frames per second of video. If 0, fps will be '
                     'read from metadata of video.') # Original:
FLAGS = flags.FLAGS

gfile = tf.io.gfile


def video_to_frames(video_filename, rotate, fps=0, resize=False,
                    width=224, height=224):
  """Returns all frames from a video.

  Args:
    video_filename: string, filename of video.
    rotate: Boolean: if True, rotates video by 90 degrees.
    fps: Integer, frames per second of video. If 0, it will be inferred from
      metadata of video.
    resize: Boolean, if True resizes images to given size.
    width: Integer, Width of image.
    height: Integer, Height of image.

  Raises:
    ValueError: if fps is greater than the rate of video.
  """
  logging.info('Loading %s', video_filename)
  cap = cv2.VideoCapture(video_filename)

  if fps == 0:
    fps = cap.get(cv2.CAP_PROP_FPS)
    keep_frequency = 1
  else:
    if fps > cap.get(cv2.CAP_PROP_FPS):
      raise ValueError('Cannot sample at a frequency higher than FPS of video')
    keep_frequency = int(float(cap.get(cv2.CAP_PROP_FPS)) / fps)

  frames = []
  timestamps = []
  counter = 0
  if cap.isOpened():
    while True:
      success, frame_bgr = cap.read()
      if not success:
        break
      if counter % keep_frequency == 0:
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        if resize:
          frame_rgb = cv2.resize(frame_rgb, (width, height))
        if rotate:
          frame_rgb = cv2.transpose(frame_rgb)
          frame_rgb = cv2.flip(frame_rgb, 1)
        frames.append(frame_rgb)
        timestamps.append(cap.get(cv2.CAP_PROP_POS_MSEC) / 1000.0)
      counter += 1
  return frames, timestamps, fps


def create_numpy(name, output_dir, input_dir, label_file, input_pattern,
                     files_per_shard, action_label, frame_labels,
                     expected_segments, orig_fps, rotate, resize, width,
                     height):
  """Create Numpy file from videos in a given path.

  Args:
    name: string, name of the dataset being created.
    output_dir: string, path to output directory.
    input_dir: string, path to input videos directory.
    label_file: None or string, JSON file that contains annotations.
    input_pattern: string, regex pattern to look up videos in directory.
    files_per_shard: int, number of files to keep in each shard.
    action_label: int, Label of actions in video.
    frame_labels: list, list of string describing each class. Class label is
      the index in list.
    expected_segments: int, expected number of segments.
    orig_fps: int, frame rate at which tfrecord will be created.
    rotate: boolean, if True rotate videos by 90 degrees.
    resize: boolean, if True resize to given height and width.
    width: int, Width of frames.
    height: int, Height of frames.
  Raises:
    ValueError: If invalid args are passed.
  """
  labels={
    'swing':0,'ball':1
  }

  ACTIVITY=FLAGS.acivity
  LABEL=labels[ACTIVITY]

  input_dir=input_dir%ACTIVITY
  output_path=output_dir%ACTIVITY

  if not gfile.exists(output_path):
    logging.info('Creating output directory: %s', output_path)
    gfile.makedirs(output_path)

  if not isinstance(input_pattern, list):
    file_pattern = os.path.join(input_dir, input_pattern)
    filenames = [os.path.basename(x) for x in gfile.glob(file_pattern)]
  else:
    filenames = []
    for file_pattern in input_pattern:
      file_pattern = os.path.join(input_dir, file_pattern)
      filenames += [os.path.basename(x) for x in gfile.glob(file_pattern)]
  
  num_shards = int(math.ceil(len(filenames)/files_per_shard))
  len_num_shards = len(str(num_shards))
  shard_id = 0

  image_minibatch=list()
  step_minibatch=list()
  label_minibatch=list()
  video_minibatch=list()

  for i, filename in enumerate(filenames):

    frames, video_timestamps, _ = video_to_frames(
        os.path.join(input_dir, filename),
        rotate,
        orig_fps,
        resize=resize,
        width=width,
        height=height)
    
    vid_name=os.path.splitext(filename)[0]
    vid_name=str.encode(vid_name)

    image_minibatch.append(frames)

    steps=np.array([math.floor(x/0.0667) for x in video_timestamps])
    step_minibatch.append(steps)

    labels=[LABEL]*len(steps)
    label_minibatch.append(labels)

    vids=[vid_name]*len(steps)
    video_minibatch+=vids

    if (i + 1) % files_per_shard == 0 or i == len(filenames) - 1:
      output_filename = os.path.join(
          output_path,
          '%s-%s-of-%s.npy' % (name,
                                    str(shard_id).zfill(len_num_shards),
                                    str(num_shards).zfill(len_num_shards)))

      image_minibatch=np.concatenate(image_minibatch,axis=0)
      step_minibatch=np.concatenate(step_minibatch,axis=0)
      label_minibatch=np.concatenate(label_minibatch,axis=0)

      numpy_dict={
          'images':image_minibatch, # np.array: B*H*W*3
          'activity':label_minibatch, # np.array: B*1
          'steps':step_minibatch, # np.array:B*1
          'videos':video_minibatch,# list
        }

      with open(output_filename,'wb') as file:
        np.save(file,numpy_dict)

      shard_id += 1
      image_minibatch=list()
      step_minibatch=list()
      label_minibatch=list()
      video_minibatch=list()


def main(_):
  create_numpy(FLAGS.name, FLAGS.output_dir, FLAGS.input_dir,
                FLAGS.label_file, FLAGS.file_pattern, FLAGS.files_per_shard,
                FLAGS.action_label, FLAGS.frame_labels,
                FLAGS.expected_segments, FLAGS.fps, FLAGS.rotate,
                FLAGS.resize, FLAGS.width, FLAGS.height)



if __name__ == '__main__':
  app.run(main)