ef create_tf_example(example):
  # TODO(user): Populate the following variables from your example.
  height = 720 # Image height
  width = 1280 # Image width
  filename = None # Filename of the image. Empty if image is not from file
  encoded_image_data = None # Encoded image bytes
  image_format = b'jpg' # b'jpeg' or b'png'

  xmins = [0]*nframes # List of normalized left x coordinates in bounding box (1 per box)
  xmaxs = [0]*nframes # List of normalized right x coordinates in bounding box
             # (1 per box)
  ymins = [0]*nframes # List of normalized top y coordinates in bounding box (1 per box)
  ymaxs = [0]*nframes # List of normalized bottom y coordinates in bounding box
             # (1 per box)
  classes_text = ['Human']*nframes # List of string class name of bounding box (1 per box)
  classes = [1]*nframes # List of integer class id of bounding box (1 per box)

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
  writer = tf.python_io.TFRecordWriter(FLAGS.output_path)

  # TODO(user): Write code to read in your dataset to examples variable
  examples=data

  for example in examples:
    tf_example = create_tf_example(example)
    writer.write(tf_example.SerializeToString())

  writer.close()


# if __name__ == '__main__':
tf.app.run()