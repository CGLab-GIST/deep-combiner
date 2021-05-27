
import os
import numpy as np
import glob

import exr as exr
import tensorflow as tf
import config as conf


def _bytes_feature(value):
    return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def parse_single_example(serialized_item):
    feat_description = {
        'x_feat': tf.FixedLenFeature([], tf.string),
        'x_corr': tf.FixedLenFeature([], tf.string),
        'x_rand': tf.FixedLenFeature([], tf.string),
        'y_ref': tf.FixedLenFeature([], tf.string),
    }
    example = tf.parse_single_example(serialized_item, features=feat_description)

    x_f = tf.reshape(tf.decode_raw(example.get('x_feat'), tf.float32), [conf.FLAGS.image_h, conf.FLAGS.image_w, conf.FLAGS.input_channel])
    x_c = tf.reshape(tf.decode_raw(example.get('x_corr'), tf.float32), [conf.FLAGS.image_h, conf.FLAGS.image_w, conf.FLAGS.color_channel])
    x_r = tf.reshape(tf.decode_raw(example.get('x_rand'), tf.float32), [conf.FLAGS.image_h, conf.FLAGS.image_w, conf.FLAGS.color_channel])
    y_ref = tf.reshape(tf.decode_raw(example.get('y_ref'), tf.float32), [conf.FLAGS.image_h, conf.FLAGS.image_w, conf.FLAGS.color_channel])

    data_dic = {'x_feat': x_f, 'x_corr': x_c, 'x_rand': x_r, 'y_ref': y_ref}

    return dict(data_dic)


def readTFRecord(fileDir, fileName, batchSize=8, numEpoch=200):
    recordSet = [fn for fn in sorted(glob.glob(os.path.join(fileDir, fileName + '*.tfrecord')))]
    
    # randomness for reading tfrecord
    np.random.seed(42)
    pIdxSet = np.random.permutation(len(recordSet))
    copiedRecordSet = recordSet.copy()
    for pIdx in range(len(recordSet)):
        copiedRecordSet[pIdx] = recordSet[pIdxSet[pIdx]]

    raw_dataset = tf.data.TFRecordDataset(recordSet)
    dataset = raw_dataset.map(lambda x : parse_single_example(x))
    dataset = dataset.shuffle(2 * batchSize)
    dataset = dataset.batch(batchSize)
    dataset = dataset.repeat(numEpoch)
    iterator = dataset.make_one_shot_iterator()

    return iterator.get_next()


def writeTFRecord(data, writer):
    features = {
        'x_feat': _bytes_feature((data.get('x_feat')).tobytes()),
        'x_corr': _bytes_feature((data.get('x_corr')).tobytes()),
        'x_rand': _bytes_feature((data.get('x_rand')).tobytes()),
        'y_ref': _bytes_feature((data.get('y_ref')).tobytes()),
    }
    ex = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(ex.SerializeToString())


def readOneFrame(framePath, targetFramePath, typeCombiner):
    # texture
    x_texture = [exr.read(framePath + '_albedo.exr')]
    x_texture = np.array(x_texture)
    x_texture = x_texture[:, :, :, 0:3]
    
    # corrImg
    x_corr_img = [exr.read(framePath + '_corrImg.exr')]
    if typeCombiner == conf.TYPE_MULTI_BUFFER:
        x_corr_img0 = [exr.read(framePath + '_corrImg_0.exr')]
        x_corr_img1 = [exr.read(framePath + '_corrImg_1.exr')]
        x_corr_img2 = [exr.read(framePath + '_corrImg_2.exr')]
        x_corr_img3 = [exr.read(framePath + '_corrImg_3.exr')]

    # depth
    x_depth = [exr.read(framePath + '_depth.exr')]
    x_depth = np.array(x_depth)
    x_depth = x_depth[:, :, :, 0:1]
    x_depth = (x_depth - np.min(x_depth)) / (np.max(x_depth) - np.min(x_depth))
    
    # normal
    x_normal = [exr.read(framePath + '_normal.exr')]
    x_normal = np.array(x_normal)
    
    # randImg
    x_rand_img = [exr.read(framePath + '_randImg.exr')]
    if typeCombiner == conf.TYPE_MULTI_BUFFER:
        x_rand_img0 = [exr.read(framePath + '_randImg_0.exr')]
        x_rand_img1 = [exr.read(framePath + '_randImg_1.exr')]
        x_rand_img2 = [exr.read(framePath + '_randImg_2.exr')]
        x_rand_img3 = [exr.read(framePath + '_randImg_3.exr')]
    
    # reference
    y_color = [exr.read(targetFramePath + '.exr')]
    y_color = np.array(y_color)
    y_color = y_color[:, :, :, 0:3]
    y_ref = np.concatenate([y_color], axis=3)

    x_corr_img = np.array(x_corr_img)
    x_corr_img = x_corr_img[:, :, :, 0:3]
    if typeCombiner == conf.TYPE_MULTI_BUFFER:
        x_corr_img0 = np.array(x_corr_img0)
        x_corr_img0 = x_corr_img0[:, :, :, 0:3]
        x_corr_img1 = np.array(x_corr_img1)
        x_corr_img1 = x_corr_img1[:, :, :, 0:3]
        x_corr_img2 = np.array(x_corr_img2)
        x_corr_img2 = x_corr_img2[:, :, :, 0:3]
        x_corr_img3 = np.array(x_corr_img3)
        x_corr_img3 = x_corr_img3[:, :, :, 0:3]

    x_rand_img = np.array(x_rand_img)
    x_rand_img = x_rand_img[:, :, :, 0:3]
    if typeCombiner == conf.TYPE_MULTI_BUFFER:
        x_rand_img0 = np.array(x_rand_img0)
        x_rand_img0 = x_rand_img0[:, :, :, 0:3]
        x_rand_img1 = np.array(x_rand_img1)
        x_rand_img1 = x_rand_img1[:, :, :, 0:3]
        x_rand_img2 = np.array(x_rand_img2)
        x_rand_img2 = x_rand_img2[:, :, :, 0:3]
        x_rand_img3 = np.array(x_rand_img3)
        x_rand_img3 = x_rand_img3[:, :, :, 0:3]

    if typeCombiner == conf.TYPE_MULTI_BUFFER:
        x_feat = np.concatenate([x_corr_img0, x_corr_img1, x_corr_img2, x_corr_img3, \
            x_rand_img0, x_rand_img1, x_rand_img2, x_rand_img3, x_normal, x_texture, x_depth], axis=3)
    elif typeCombiner == conf.TYPE_SINGLE_BUFFER:
        x_feat = np.concatenate([x_corr_img, x_rand_img, x_normal, x_texture, x_depth], axis=3)
    x_corr = np.concatenate([x_corr_img], axis=3)
    x_rand = np.concatenate([x_rand_img], axis=3)

    dataList = {}
    dataList['x_feat'] = x_feat
    dataList['x_corr'] = x_corr
    dataList['x_rand'] = x_rand
    dataList['y_ref'] = y_ref

    return dataList


def generateTrainingData(FLAGS):
    if not os.path.exists(FLAGS.train_dataset_dir):
        os.makedirs(FLAGS.train_dataset_dir)

    fileIdx = 0
    totalFrame = 20
    permIdxSet = np.random.permutation(totalFrame)

    for spp in FLAGS.train_spps:
        for corrMethod in FLAGS.train_corr_methods:
            for scene in FLAGS.train_scenes:
                strPathInput = os.path.join(FLAGS.train_input_dir, scene + '_' + corrMethod + '_' + spp)
                strPathTarget = os.path.join(FLAGS.train_target_dir, scene)

                for fIdx in range(0, totalFrame):
                    frameIdx = permIdxSet[fIdx]
                    strFileIdx = '{0:03d}'.format(fileIdx)
                    strFrameIdx = '{0:03d}'.format(frameIdx)
                    setPathInput = [fn for fn in sorted(glob.glob(os.path.join(strPathInput, strFrameIdx + '*.exr')))]
                    if len(setPathInput) == 0:
                        print("[image_io.py] Not found dataset at frame index %d from %s scene. Skip it" % (frameIdx, scene))
                        continue

                    print("[image_io.py] %d-th... Working on frame index %d in %s, (%s spp, %s)..." % (fileIdx, frameIdx, scene, spp, corrMethod))
                    strPathOutput = os.path.join(FLAGS.train_dataset_dir, FLAGS.tfrecord_filename + strFileIdx + '.tfrecord')
                    writer = tf.python_io.TFRecordWriter(strPathOutput)
                    currInputFramePath = os.path.join(strPathInput, strFrameIdx)
                    currTargetFramePath = os.path.join(strPathTarget, strFrameIdx)
                    data = readOneFrame(currInputFramePath, currTargetFramePath, FLAGS.type_combiner)
                    writeTFRecord(data, writer)
                    writer.close()
                    fileIdx += 1


    print("[image_io.py] Total %d frames are used in dataset!" % (fileIdx))

