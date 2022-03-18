import os
import glob
import time
from datetime import datetime
import numpy as np
import tensorflow.compat.v1 as tf

import exr as exr
import model as model
import loss as loss
import image_io as imgIo
import config as conf

class Network:
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

    def __init__(self, _h, _w, _b, _flags):
        self.step = 0
        self.FLAGS = _flags
        
        tf.set_random_seed(_flags.train_seed)
        self.is_training = tf.placeholder(tf.bool)
        self.TENSOR_SHAPE_INPUT_FEAT = [_b, _h, _w, _flags.input_channel]
        self.TENSOR_SHAPE_OUTPUT = [_b, _h, _w, _flags.color_channel]

        self.loaded_frames = imgIo.readTFRecord(_flags.train_dataset_dir, _flags.tfrecord_filename, _b, _flags.num_epoch)

        self.x = tf.placeholder(tf.float32, self.TENSOR_SHAPE_INPUT_FEAT, name='x')
        self.y = tf.placeholder(tf.float32, self.TENSOR_SHAPE_OUTPUT, name='y')
        if _flags.type_combiner == conf.TYPE_MULTI_BUFFER:
            self._y = model.COMBINER_MULTI_BUFFER(self.x, _flags.kernel_size, _h, _w, _b, self.is_training)
        elif _flags.type_combiner == conf.TYPE_SINGLE_BUFFER:
            self._y = model.COMBINER_SINGLE_BUFFER(self.x, _flags.kernel_size, _h, _w, _b, self.is_training)
        
        self.loss = loss.RELMSE(self._y, self.y)
        self.loss_op = loss.minimizeAdamOptimizer(_flags.learning_rate, self.loss, name='adam')


    def train(self, retrain=False, validate=False, valid_stamp=1):
        print("[network.py] Train started!")

        recordSet = glob.glob(os.path.join(self.FLAGS.train_dataset_dir, \
            self.FLAGS.tfrecord_filename + '*.tfrecord'))
        itersPerEpoch = int(len(recordSet) / self.FLAGS.batch_size)
        if len(recordSet) == 0:
            print("[network.py] There is no tfrecord file. Please check it!")
            return

        tf_config = tf.ConfigProto() 
        tf_config.gpu_options.allow_growth = True

        with tf.Session(config=tf_config) as sess:
            sess.run(tf.global_variables_initializer())
            logger = Logger(sess, self.FLAGS.logger_dir, self.FLAGS.ckpt_dir, reset=False)
            summary = logger.summary(self.loss)
            minValLoss = float("inf")
            
            if (retrain):
                logger.restore_latest_ckpt(sess)
                print("[network.py] Restored network parameters!")
        
            for epoch in range(self.FLAGS.num_epoch):
                f = open(self.FLAGS.data_dir + 'log_train_loss.txt', 'a')
                startTime = time.time()
                currTrainLoss = 0
                currTrainIter = 0
                for iters in range(itersPerEpoch):
                    loadedFrames = sess.run(self.loaded_frames)
                    x_feat = loadedFrames['x_feat']
                    y_ref = loadedFrames['y_ref']

                    index = self.get_index(x_feat)
                    numDividedPatches = int(index.shape[0] / self.FLAGS.batch_size)

                    for step in range(numDividedPatches):
                        #startTime = time.time()
                        batch_x, batch_y = self.get_next_batch(step, index, x_feat, y_ref)

                        feed = {self.y: batch_y, self.x: batch_x, self.is_training: True}
                        _, loss, summary_loss, train_output = sess.run([self.loss_op, self.loss, summary, self._y], feed_dict=feed)
                        
                        logger.update_loss(loss, summary_loss, epoch, self.step)
                        currTrainLoss += loss
                        currTrainIter += 1
                        self.step += 1
                        #endTime = time.time() - startTime
                        #print("[network.py] Running time : %f sec" % (endTime))

                currTrainLoss /= float(currTrainIter)
                endTime = time.time() - startTime
                strDate = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                strPrintLoss = "[network.py] epoch : %d (%f sec, %s) | training loss : %f\n" % (epoch, endTime, strDate, currTrainLoss)
                print(strPrintLoss)
                f.write(strPrintLoss)
                f.close()
                        

                # Validation
                if epoch % valid_stamp == 0:
                    logger.save_ckpt(sess, epoch)
                    if (validate):
                        f = open(self.FLAGS.data_dir + 'log_validate_loss.txt', 'a')
                        currValLoss = 0
                        currValIter = 0
                        valRecordSet = glob.glob(os.path.join(self.FLAGS.valid_dataset_dir, \
                            self.FLAGS.tfrecord_filename + '*.tfrecord'))
                        valItersPerEpoch = int(len(valRecordSet) / self.FLAGS.batch_size)
                        if len(valRecordSet) == 0:
                            print("[network.py] There is no tfrecord file for validation. skip it!")
                        else:
                            valLoadedFrames = imgIo.readTFRecord(self.FLAGS.valid_dataset_dir, self.FLAGS.tfrecord_filename, self.FLAGS.batch_size, self.FLAGS.num_epoch)
                            for valIters in range(valItersPerEpoch):
                                valLoadedFrame = sess.run(valLoadedFrames)
                                v_x_feat = valLoadedFrame['x_feat']
                                v_y_ref = valLoadedFrame['y_ref']

                                v_index = self.get_index(v_x_feat)
                                numDividedValPatches = int(v_index.shape[0] / self.FLAGS.batch_size)

                                for v_step in range(numDividedValPatches):
                                    b_x, b_y = self.get_next_batch(v_step, v_index, v_x_feat, v_y_ref)

                                    feed = {self.y: b_y, self.x: b_x, self.is_training: False}
                                    v_loss = sess.run(self.loss, feed_dict=feed)

                                    currValLoss += v_loss
                                    currValIter += 1


                        currValLoss /= float(currValIter)
                        strDate = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                        strPrintLoss = "[network.py] epoch : %d (%s) | validate loss : %f\n" % (epoch, strDate, currValLoss)
                        print(strPrintLoss)
                        f.write(strPrintLoss)
                        f.close()

                        if minValLoss > currValLoss:
                            logger.save_optimal_ckpt(sess, epoch)
                            minValLoss = currValLoss


    def test(self):
        print("[network.py] Test started!")
        if not os.path.exists(self.FLAGS.test_output_dir):
            os.makedirs(self.FLAGS.test_output_dir)
        
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
        
            logger = Logger(sess, self.FLAGS.logger_dir, self.FLAGS.ckpt_dir, reset=False)
            logger.restore_ckpt(sess, self.FLAGS.test_ckpt_filename)
            
            METHODS = self.FLAGS.test_corr_methods
            TEST_SCENES = self.FLAGS.test_scenes
            TEST_SPPS = self.FLAGS.test_spps
            TOTAL_FRAMES = self.FLAGS.test_frame_num
            
            for spp in TEST_SPPS:
                for testMethod in METHODS:
                    for testScene in TEST_SCENES:
                        strInputDir = os.path.join(self.FLAGS.test_input_dir, testScene + '_' + testMethod + '_' + spp)
                        strTargetDir = os.path.join(self.FLAGS.test_target_dir, testScene)
                        strOutDir = os.path.join(self.FLAGS.test_output_dir, testScene + '_' + testMethod + '_' + spp)
                        if not os.path.exists(strOutDir):
                            os.makedirs(strOutDir)
                
                        for testIters in range(TOTAL_FRAMES):
                            strFrameIdx = '{0:03d}'.format(testIters)
                            inputFrames = [fn for fn in sorted(glob.glob(os.path.join(strInputDir, strFrameIdx + '*.exr')))]
                            targetFrame = [fn for fn in sorted(glob.glob(os.path.join(strTargetDir, '000.exr')))]

                            if len(inputFrames) == 0 or len(targetFrame) == 0:
                                print("[network.py] Please prepare for test dataset at frame index %d in %s. Skip it" % (testIters, testScene))
                                continue

                            print("[network.py] %d-th frame from (%s, %sspp)..." % (testIters, testScene, spp))
                            strCurrInputDir = os.path.join(strInputDir, strFrameIdx)
                            strCurrTargetDir = os.path.join(strTargetDir, strFrameIdx)
                            strCurrOutDir = os.path.join(strOutDir, strFrameIdx)

                            test_x = imgIo.readOneFrame(strCurrInputDir, strCurrTargetDir, self.FLAGS.type_combiner)

                            t_x = test_x['x_feat']
                            t_x_corrImgs = test_x['x_corr']
                            t_x_randImgs = test_x['x_rand']
                            t_y = test_x['y_ref']

                            startTime = time.time()

                            feed = {self.x: t_x, self.y: t_y, self.is_training: False}
                            out_y = sess.run(self._y, feed_dict=feed)

                            endTime = time.time() - startTime
                            print("[network.py] Running time : %f sec" % (endTime))

                            in_cs = np.clip(t_x_corrImgs[0], 0, None)
                            in_rs = np.clip(t_x_randImgs[0], 0, None)
                            ref_y = np.array(t_y[0])
                            out_y = np.clip(out_y[0], 0, None)

                            relMSE_cs = self.get_relMSE(in_cs, ref_y)
                            relMSE_rs = self.get_relMSE(in_rs, ref_y)
                            relMSE_out = self.get_relMSE(out_y, ref_y)

                            print("[network.py] relMSE of corrImg : %f" % relMSE_cs)
                            print("[network.py] relMSE of randImg : %f" % relMSE_rs)
                            print("[network.py] relMSE of outImg  : %f" % relMSE_out)

                            exr.write(strCurrOutDir + '_in_corrImg_' + '{:.6f}'.format(relMSE_cs) + '.exr', in_cs)
                            exr.write(strCurrOutDir + '_in_randImg_' + '{:.6f}'.format(relMSE_rs) + '.exr', in_rs)
                            exr.write(strCurrOutDir + '_out_' + '{:.6f}'.format(relMSE_out) + '.exr', out_y)


    def get_index(self, x, stride=50):
        index=[]
        for i in range(len(x)):
            h, w, _ = x[i].shape
            for iy in range(0, h - self.FLAGS.patch_size_pad + 1, stride):
                for ix in range(0, w - self.FLAGS.patch_size_pad + 1, stride):
                    index.append([i, iy, ix])
    
        index = np.array(index)
        s = np.arange(index.shape[0])
        np.random.shuffle(s)

        if self.FLAGS.dataset_sampling_on:
            sampled_num = int(len(s) * self.FLAGS.dataset_sampling_rate)
            s = s[0:sampled_num]
    
        return index[s]


    def get_next_batch(self, batch_idx, index, x, y):
        batch_size = self.FLAGS.batch_size
        patch_size = self.FLAGS.patch_size
        half_pad_size = self.FLAGS.half_kernel_size
        
        batch_x = []
        batch_y = []

        for b in range(batch_size):    
            fi = index[batch_size * batch_idx + b][0]
            patch_hi = index[batch_size * batch_idx + b][1]
            patch_wi = index[batch_size * batch_idx + b][2]

            sequence_x = x[fi][patch_hi+half_pad_size: patch_hi+half_pad_size+patch_size, patch_wi+half_pad_size: patch_wi+half_pad_size+patch_size, :]
            sequence_y = y[fi][patch_hi+half_pad_size: patch_hi+half_pad_size+patch_size, patch_wi+half_pad_size: patch_wi+half_pad_size+patch_size, :]

            batch_x.append(sequence_x)
            batch_y.append(sequence_y)

        return np.array(batch_x), np.array(batch_y)


    def get_relMSE(self, input, ref):
        h, w, _ = np.shape(input)
        num = np.square(np.subtract(input, ref))
        denom = np.mean(ref, axis=2)
        denom = np.reshape(denom, (h, w, 1))
        relMSE = num / (denom * denom + 1e-2)
        relMSEMean = np.mean(relMSE)

        return relMSEMean


class Logger:

    def __init__(self, sess, logger_dir, ckpt_dir, train=True, reset=True):
        self.logger_dir = logger_dir
        self.ckpt_dir = ckpt_dir

        self.ckpt_saver = tf.train.Saver()
        self.ckpt_opt_saver = tf.train.Saver()
        self.train_logger = tf.summary.FileWriter(self.logger_dir, sess.graph)

    def summary(self, loss):
        tf.summary.scalar("loss", loss)
        return tf.summary.merge_all()

    def update_loss(self, loss, summary_loss, e, step):
        if (step % 1 == 0):
            print("epoch: %d, step: %2d, loss: %.8f" % (e, (step+1), loss))
        self.train_logger.add_summary(summary_loss, step)

    def save_ckpt(self, sess, epoch):
        self.ckpt_saver.save(sess, self.ckpt_dir + "/tr_model_epoch" + str(epoch) +".ckpt", write_meta_graph=False)

    def save_optimal_ckpt(self, sess, epoch):
        if not os.path.exists(self.ckpt_dir + "/optimal"):
            os.makedirs(self.ckpt_dir + "/optimal")
        self.ckpt_opt_saver.save(sess, self.ckpt_dir + "/optimal/tr_optimal_model_epoch" + str(epoch) +".ckpt", write_meta_graph=False)

    def restore_ckpt(self, sess, ckpt_name):
        self.ckpt_saver.restore(sess, self.ckpt_dir + "/" + ckpt_name)

    def restore_latest_ckpt(self, sess):
        self.ckpt_saver.restore(sess, tf.train.latest_checkpoint(self.ckpt_dir))
