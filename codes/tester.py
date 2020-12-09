
import config
import image_io
from network import Network

FLAGS = config.FLAGS

if FLAGS.mode_setting == config.MODE_DATA_GENERATION:
    image_io.generateTrainingData(FLAGS)

elif FLAGS.mode_setting == config.MODE_TRAIN:
    network = Network(FLAGS.patch_size, FLAGS.patch_size, FLAGS.batch_size, FLAGS)
    network.train(retrain=FLAGS.retraining_on, validate=FLAGS.validation_on)

elif FLAGS.mode_setting == config.MODE_TEST:
    network = Network(FLAGS.image_h, FLAGS.image_w, 1, FLAGS)
    network.test()

print('Done!')
