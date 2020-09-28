# """
run_ssd_inception_v2.py

# nets

export PYTHONPATH=$PYTHONPATH:/home/felicia/research/models-1.13.0/research/:/home/felicia/research/models-1.13.0/research/slim



PIPELINE_CONFIG_PATH={path to pipeline config file}
MODEL_DIR={path to model directory}
NUM_TRAIN_STEPS=50000
SAMPLE_1_OF_N_EVAL_EXAMPLES=1
python object_detection/model_main.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr

nframes=1070

PIPELINE_CONFIG_PATH='/home/felicia/research/models-1.13.0/research/object_detection/samples/configs/ssd_inception_v2_mlb.config'
MODEL_DIR='/home/felicia/research/models-1.13.0/research/object_detection/saved_models/'
NUM_TRAIN_STEPS=1
SAMPLE_1_OF_N_EVAL_EXAMPLES=1070
# CHECKPOINT='/home/felicia/research/models-1.13.0/research/object_detection/pretrained/'
python object_detection/run_ssd_inception_v2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr

    # --checkpoint_dir=${CHECKPOINT}\



PIPELINE_CONFIG_PATH='/home/felicia/research/models-1.13.0/research/object_detection/samples/configs/ssd_inception_v2_coco_test.config'
MODEL_DIR='/home/felicia/research/models-1.13.0/research/object_detection/saved_models/'
NUM_TRAIN_STEPS=1
SAMPLE_1_OF_N_EVAL_EXAMPLES=100
# CHECKPOINT='/home/felicia/research/models-1.13.0/research/object_detection/pretrained/'
python object_detection/run_ssd_inception_v2.py \
    --pipeline_config_path=${PIPELINE_CONFIG_PATH} \
    --model_dir=${MODEL_DIR} \
    --num_train_steps=${NUM_TRAIN_STEPS} \
    --sample_1_of_n_eval_examples=$SAMPLE_1_OF_N_EVAL_EXAMPLES \
    --alsologtostderr

    # --checkpoint_dir=${CHECKPOINT}\


--------------------------------------------------------------------------------------------------------------------------------

1. object_detection/utils/label_mao_util.py:
def get_label_map_dict(label_map_path,
                       use_display_name=False,
                       fill_in_gaps_and_background=False):
# line 164 
label_map = load_labelmap(label_map_path)

# label_map_path

2. object_detection/data_decoders/tf_example_decoder.py:

class TfExampleDecoder(data_decoder.DataDecoder):
  """Tensorflow Example proto decoder."""

  def __init__(self,
               load_instance_masks=False,
               instance_mask_type=input_reader_pb2.NUMERICAL_MASKS,
               label_map_proto_file=None,
               use_display_name=False,
               dct_method='',
               num_keypoints=0,
               num_additional_channels=0):

# line 297 
label_handler = _BackupHandler(
    _ClassTensorHandler(
        'image/object/class/text', label_map_proto_file,
        default_value=''),
slim_example_decoder.Tensor('image/object/class/label'))

class _ClassTensorHandler(slim_example_decoder.Tensor):
  """An ItemHandler to fetch class ids from class text."""

  def __init__(self,
               tensor_key,
               label_map_proto_file,
               shape_keys=None,
               shape=None,
               default_value=''):

# line 59 
name_to_id = label_map_util.get_label_map_dict(
        label_map_proto_file, use_display_name=False)
    
# label_map_proto_file

3. object_detection/builders/dataset_builder.py

def build(input_reader_config, batch_size=None, transform_input_data_fn=None):


label_map_proto_file = None
if input_reader_config.HasField('label_map_path'):
    label_map_proto_file = input_reader_config.label_map_path
# line 123     
decoder = tf_example_decoder.TfExampleDecoder(
        load_instance_masks=input_reader_config.load_instance_masks,
        instance_mask_type=input_reader_config.mask_type,
        label_map_proto_file=label_map_proto_file,
        use_display_name=input_reader_config.use_display_name,
        num_additional_channels=input_reader_config.num_additional_channels)

# label_map_proto_file
# input_reader_config

4. research/object_detection/inputs.py:

# line 44
INPUT_BUILDER_UTIL_MAP = {
    'dataset_build': dataset_builder.build,
}

def create_eval_input_fn(eval_config, eval_input_config, model_config):

# line 579
dataset = INPUT_BUILDER_UTIL_MAP['dataset_build'](
    eval_input_config,
    batch_size=params['batch_size'] if params else eval_config.batch_size,
    transform_input_data_fn=transform_and_pad_input_data_fn)

# eval_input_config --> input_reader_config


5. object_detection/model_lib.py
MODEL_BUILD_UTIL_MAP = {
    'get_configs_from_pipeline_file':
        config_util.get_configs_from_pipeline_file,
    'create_pipeline_proto_from_configs':
        config_util.create_pipeline_proto_from_configs,
    'merge_external_params_with_configs':
        config_util.merge_external_params_with_configs,
    'create_train_input_fn':
        inputs.create_train_input_fn,
    'create_eval_input_fn':
        inputs.create_eval_input_fn,
    'create_predict_input_fn':
        inputs.create_predict_input_fn,
}

def create_estimator_and_inputs(run_config,
                                hparams,
                                pipeline_config_path,
                                config_override=None,
                                train_steps=None,
                                sample_1_of_n_eval_examples=1,
                                sample_1_of_n_eval_on_train_examples=1,
                                model_fn_creator=create_model_fn,
                                use_tpu_estimator=False,
                                use_tpu=False,
                                num_shards=1,
                                params=None,
                                override_eval_num_epochs=True,
                                save_final_config=False,
                                **kwargs):

    eval_on_train_input_fn = create_eval_input_fn(
        eval_config=eval_config,
        eval_input_config=eval_on_train_input_config,
        model_config=model_config)

6. object_detection/run_ssd_inception_v2.py：
def main(unused_argv):

train_and_eval_dict = model_lib.create_estimator_and_inputs(
    run_config=config,
    hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
    pipeline_config_path=FLAGS.pipeline_config_path,
    train_steps=FLAGS.num_train_steps,
    sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
    sample_1_of_n_eval_on_train_examples=(
        FLAGS.sample_1_of_n_eval_on_train_examples))
estimator = train_and_eval_dict['estimator']

eval_on_train_input_fn = train_and_eval_dict['eval_on_train_input_fn']

if FLAGS.checkpoint_dir:
    if FLAGS.eval_training_data:
      name = 'training_data'
      input_fn = eval_on_train_input_fn
    else:
      name = 'validation_data'
      # The first eval input will be evaluated.
      input_fn = eval_input_fns[0]
    if FLAGS.run_once:
    #   estimator.evaluate(input_fn,
    #                      num_eval_steps=None,
    #                      checkpoint_path=tf.train.latest_checkpoint(
    #                          FLAGS.checkpoint_dir))
      estimator.evaluate(input_fn,
                         checkpoint_path=tf.train.latest_checkpoint(
                             FLAGS.checkpoint_dir))
# line 89
    else:
      model_lib.continuous_eval(estimator, FLAGS.checkpoint_dir, input_fn,
                                train_steps, name)
  else:
    train_spec, eval_specs = model_lib.create_train_and_eval_specs(
        train_input_fn,
        eval_input_fns,
        eval_on_train_input_fn,
        predict_input_fn,
        train_steps,
        eval_on_train_data=False)

############
FLAGS.checkpoint_dir
input_fn -> eval_on_train_input_fn -> train_and_eval_dict['eval_on_train_input_fn']: An evaluation-on-train input function.
FLAGS.checkpoint_dir: CHECKPOINT='/home/felicia/research/models-1.13.0/research/object_detection/pretrained/'

----------------------------------------
run_ssd_inception_v2 -> model_lib -> inputs-> dataset_builder -> tf_example_decoder -> label_map_util


----------------------'hyperparams_config'----------------------------------
1. object_detection/builders/hyperparams_builder.py

def build(hyperparams_config, is_training):
    """
    The batch norm parameters are set for updates based on `is_training` argument
    and conv_hyperparams_config.batch_norm.train parameter. During training, they
    are updated only if batch_norm.train parameter is true. However, during eval,
    no updates are made to the batch norm variables. In both cases, their current
    values are used during forward pass.

    Args:
    hyperparams_config: hyperparams.proto object containing
        hyperparameters.
    is_training: Whether the network is in training mode.
    """

# line 213
    if not isinstance(hyperparams_config,
                    hyperparams_pb2.Hyperparams):
        raise ValueError('hyperparams_config not of type '
                            'hyperparams_pb.Hyperparams.')

# hyperparams_config

2. object_detection/builders/model_builder.py

# line 157
def build(model_config, is_training, add_summaries=True):

    if meta_architecture == 'ssd':
        return _build_ssd_model(model_config.ssd, is_training, add_summaries)

# line 274
def _build_ssd_model(ssd_config, is_training, add_summaries):

    feature_extractor = _build_ssd_feature_extractor(
        feature_extractor_config=ssd_config.feature_extractor,
        freeze_batchnorm=ssd_config.freeze_batchnorm,
        is_training=is_training)  

# line 198 
def _build_ssd_feature_extractor(feature_extractor_config,
                                 is_training,
                                 freeze_batchnorm,
                                 reuse_weights=None):

  if is_keras_extractor:
    conv_hyperparams = hyperparams_builder.KerasLayerHyperparams(
        feature_extractor_config.conv_hyperparams)
  else: # False
    conv_hyperparams = hyperparams_builder.build(
        feature_extractor_config.conv_hyperparams, is_training)

# is_keras_extractor=False

#########
model_builder.build --model_config.ssd.feature_extractor.conv_hyperparams-> model_builder._build_ssd_model --ssd_config.feature_extractor.conv_hyperparams-> model_builder._build_ssd_feature_extractor --feature_extractor_config.conv_hyperparams-> hyperparams_builder.build --hyperparams_config ->


3. object_detection/builders/dataset_builder.py
def build(input_reader_config, batch_size=None, transform_input_data_fn=None):
# line 129
    if transform_input_data_fn is not None:
        processed_tensors = transform_input_data_fn(processed_tensors)



4. object_detection/inputs.py
def create_train_input_fn(train_config, train_input_config,
                          model_config):
"""
 model_config: A model_pb2.DetectionModel
"""
# line 465
    model = model_builder.build(model_config, is_training=True)
------------------------------------
inputs.create_train_input_fn --model_config->model_builder.build-> hyperparams_builder


------------------------------------
inputs-> dataset_builder-> 

5. object_detection/model_lib.py

MODEL_BUILD_UTIL_MAP = {
    'get_configs_from_pipeline_file':
        config_util.get_configs_from_pipeline_file,
    'create_pipeline_proto_from_configs':
        config_util.create_pipeline_proto_from_configs,
    'merge_external_params_with_configs':
        config_util.merge_external_params_with_configs,
    'create_train_input_fn':
        inputs.create_train_input_fn,
    'create_eval_input_fn':
        inputs.create_eval_input_fn,
    'create_predict_input_fn':
        inputs.create_predict_input_fn,
}


# line 576
    create_train_input_fn = MODEL_BUILD_UTIL_MAP['create_train_input_fn']

def create_estimator_and_inputs(run_config,
                                hparams,
                                pipeline_config_path,
                                config_override=None,
                                train_steps=None,
                                sample_1_of_n_eval_examples=1,
                                sample_1_of_n_eval_on_train_examples=1,
                                model_fn_creator=create_model_fn,
                                use_tpu_estimator=False,
                                use_tpu=False,
                                num_shards=1,
                                params=None,
                                override_eval_num_epochs=True,
                                save_final_config=False,
                                **kwargs):
# line 570
    get_configs_from_pipeline_file = MODEL_BUILD_UTIL_MAP[
        'get_configs_from_pipeline_file']
# line 580
    configs = get_configs_from_pipeline_file(pipeline_config_path,
                                            config_override=config_override)
# line 590
    configs = merge_external_params_with_configs(
        configs, hparams, kwargs_dict=kwargs)
# line 592   
    model_config = configs['model']
# line 616
    train_input_fn = create_train_input_fn(
        train_config=train_config,
        train_input_config=train_input_config,
        model_config=model_config)
    
    return dict(
        estimator=estimator,
        train_input_fn=train_input_fn,
        eval_input_fns=eval_input_fns,
        eval_input_names=eval_input_names,
        eval_on_train_input_fn=eval_on_train_input_fn,
        predict_input_fn=predict_input_fn,
        train_steps=train_steps)


----------------------------------------
run_ssd_inception_v2 --hparams,pipeline_config_path->model_lib.create_estimator_and_inputs --model_config-> inputs.create_train_input_fn ->model_builder.build-> hyperparams_builder


6.object_detection/utils/config_util.py
def get_configs_from_pipeline_file(pipeline_config_path, config_override=None):
# line 94
    pipeline_config = pipeline_pb2.TrainEvalPipelineConfig()
    with tf.gfile.GFile(pipeline_config_path, "r") as f:
        proto_str = f.read()
        text_format.Merge(proto_str, pipeline_config)
        
    return create_configs_from_pipeline_proto(pipeline_config)


def create_configs_from_pipeline_proto(pipeline_config):
  """Creates a configs dictionary from pipeline_pb2.TrainEvalPipelineConfig.

  Args:
    pipeline_config: pipeline_pb2.TrainEvalPipelineConfig proto object.

  Returns:
    Dictionary of configuration objects. Keys are `model`, `train_config`,
      `train_input_config`, `eval_config`, `eval_input_configs`. Value are
      the corresponding config objects or list of config objects (only for
      eval_input_configs).
  """
    configs = {}
    configs["model"] = pipeline_config.model
    configs["train_config"] = pipeline_config.train_config
    configs["train_input_config"] = pipeline_config.train_input_reader
    configs["eval_config"] = pipeline_config.eval_config
    configs["eval_input_configs"] = pipeline_config.eval_input_reader
    # Keeps eval_input_config only for backwards compatibility. All clients should
    # read eval_input_configs instead.
    if configs["eval_input_configs"]:
        configs["eval_input_config"] = configs["eval_input_configs"][0]
    if pipeline_config.HasField("graph_rewriter"):
        configs["graph_rewriter_config"] = pipeline_config.graph_rewriter

    return configs


7. object_detection/model_hparams.py

def create_hparams(hparams_overrides=None):
# line 37
    hparams = tf.contrib.training.HParams(
        # Whether a fine tuning checkpoint (provided in the pipeline config)
        # should be loaded for training.
        load_pretrained=True)

8.object_detection/run_ssd_inception_v2.py

FLAGS.pipeline_config_path='/home/felicia/research/models-1.13.0/research/object_detection/samples/configs/ssd_inception_v2_mlb.config'
FLAGS.hparams_overrides=None

def main(unused_argv):
# line 59
  train_and_eval_dict = model_lib.create_estimator_and_inputs(
      run_config=config,
      hparams=model_hparams.create_hparams(FLAGS.hparams_overrides),
      pipeline_config_path=FLAGS.pipeline_config_path,
      train_steps=FLAGS.num_train_steps,
      sample_1_of_n_eval_examples=FLAGS.sample_1_of_n_eval_examples,
      sample_1_of_n_eval_on_train_examples=(
          FLAGS.sample_1_of_n_eval_on_train_examples))


run_ssd_inception_v2 --hparams,pipeline_config_path->model_lib.create_estimator_and_inputs --model_config-> inputs.create_train_input_fn ->model_builder.build-> hyperparams_builder
--FLAGS.pipeline_config_path-> 
    

--FLAGS.hparams_overrides=None -> model_hparams.create_hparams
# config_util.get_configs_from_pipeline_file->model_lib.create_estimator_and_inputs
# """


