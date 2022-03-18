import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

FLAGS = tf.app.flags.FLAGS

""" Common config
"""
MODE_DATA_GENERATION = 0
MODE_TRAIN = 1
MODE_TEST = 2

TYPE_SINGLE_BUFFER = 0
TYPE_MULTI_BUFFER = 1

tf.app.flags.DEFINE_integer ('mode_setting', MODE_TEST,
                            'Mode selection (MODE_DATA_GENERATION, MODE_TRAIN, MODE_TEST)')
tf.app.flags.DEFINE_integer ('type_combiner', TYPE_MULTI_BUFFER,
                            'Type selection for combiner (TYPE_SINGLE_BUFFER, TYPE_MULTI_BUFFER)')

tf.app.flags.DEFINE_integer ('half_kernel_size', 7,
                            'The size of half of kernel_size_sqr (if 7 is fixed, (2*7 + 1)x(2*7 + 1) will be used.)')
tf.app.flags.DEFINE_integer ('kernel_size_sqr', (2 * FLAGS.half_kernel_size + 1),
                            'The size of side of kernel (DO NOT CHANGED)')
tf.app.flags.DEFINE_integer ('kernel_size', (FLAGS.kernel_size_sqr * FLAGS.kernel_size_sqr),
                            'Total kernel size (DO NOT CHANGED)')

multi_buffer_size = 1
if FLAGS.type_combiner == TYPE_MULTI_BUFFER:
    multi_buffer_size = 4   # JH: default is 4 (not supported for others)

tf.app.flags.DEFINE_integer ('multi_buffer_size', multi_buffer_size,
                            'The size of multi-buffer')
tf.app.flags.DEFINE_integer ('color_channel', 3,
                            'Dimension of color image (default : 3 (RGB))')
tf.app.flags.DEFINE_integer ('feature_channel', 7,
                            'Dimension of input feature (default : 7 (normal, albedo and depth))')
tf.app.flags.DEFINE_integer ('input_channel', (FLAGS.feature_channel + FLAGS.multi_buffer_size * 2 * FLAGS.color_channel),
                            'Dimension of input buffers (DO NOT CHANGED)')


""" Data-generation config
"""
tf.app.flags.DEFINE_multi_string ('train_spps', ['064'],
                            'The spp to be trained')
tf.app.flags.DEFINE_multi_string ('train_scenes', ['bathroom'],
                            'The scenes to be trained')
tf.app.flags.DEFINE_multi_string ('train_corr_methods', ['nfor'],
                            'The correlated pixel estimates to be trained')
tf.app.flags.DEFINE_string ('tfrecord_filename', 'dataset',
                            'File name of training dataset')


""" Train config
"""
tf.app.flags.DEFINE_boolean ('retraining_on', False,
                            'Choose retraining using the existing checkpoint')
tf.app.flags.DEFINE_boolean ('validation_on', True,
                            'Choose retraining or not')
tf.app.flags.DEFINE_integer ('patch_size', 128,
                            'Patch size for training')
tf.app.flags.DEFINE_integer ('patch_size_pad', (FLAGS.patch_size + 2 * FLAGS.half_kernel_size),
                            'Padded patch size for training')
tf.app.flags.DEFINE_integer ('batch_size', 10,
                            'Batch size for training')
tf.app.flags.DEFINE_integer ('num_epoch', 50,
                            'Epoch')
tf.app.flags.DEFINE_float   ('learning_rate', 1e-4,
                            'Learning rate')
tf.app.flags.DEFINE_boolean ('dataset_sampling_on', True,
                            'Random sampling of dataset for reducing learning time')
tf.app.flags.DEFINE_float   ('dataset_sampling_rate', 0.1,
                            'Rate of dataset sampling')
tf.app.flags.DEFINE_integer ('train_seed', 3233647,
                            'Random seed for training')


""" Test config
"""
tf.app.flags.DEFINE_integer ('image_h', 720,
                            'The height of testing image')
tf.app.flags.DEFINE_integer ('image_w', 1280,
                            'The width of testing image')
tf.app.flags.DEFINE_integer ('test_frame_num', 1,
                            'The number of frames with difference seeds to be tested')
tf.app.flags.DEFINE_multi_string ('test_spps', ['256'],
                            'The spp to be tested')
tf.app.flags.DEFINE_multi_string ('test_scenes', ['conference'],
                            'The scenes to be tested')                      
tf.app.flags.DEFINE_multi_string ('test_corr_methods', ['gpt_L1', 'gpt_L2', 'nfor', 'kpcn'],
                            'The correlated pixel estimates to be tested')
tf.app.flags.DEFINE_string ('test_ckpt_filename', 'opt_model_multi.ckpt',
                            'File name of checkpoint to be restored for test')


""" Directory config
"""
tf.app.flags.DEFINE_string ('data_dir', '../data/',
                            'Directory for overall dataset')

tf.app.flags.DEFINE_string ('test_input_dir', FLAGS.data_dir + '__test_scenes__/input',
                            'Directory for input buffers to be tested')
tf.app.flags.DEFINE_string ('test_target_dir', FLAGS.data_dir + '__test_scenes__/target',
                            'Directory for reference images to be tested')
tf.app.flags.DEFINE_string ('test_output_dir', FLAGS.data_dir + '__test_output__',
                            'Directory for tested output')
tf.app.flags.DEFINE_string ('train_input_dir', FLAGS.data_dir + '__train_scenes__/input',
                            'Directory for input buffers to be trained')
tf.app.flags.DEFINE_string ('train_target_dir', FLAGS.data_dir + '__train_scenes__/target',
                            'Directory for reference images to be trained')
tf.app.flags.DEFINE_string ('train_dataset_dir', FLAGS.data_dir + '__data_tfrecord__',
                            'Directory for training dataset')
tf.app.flags.DEFINE_string ('valid_dataset_dir', FLAGS.data_dir + '__valid_data_tfrecord__',
                            'Directory for training dataset')
tf.app.flags.DEFINE_string ('logger_dir', FLAGS.data_dir + '__train_log__',
                            'Directory for log')
tf.app.flags.DEFINE_string ('ckpt_dir', FLAGS.data_dir + '__train_ckpt__',
                            'Directory for checkpoint')