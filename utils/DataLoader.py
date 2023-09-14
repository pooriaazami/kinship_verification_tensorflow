import tensorflow as tf
import numpy as np

from functools import partial

def decode_image(img):
    img = tf.io.decode_jpeg(img, channels=3)
    img = tf.cast(img, tf.float32)
    img = tf.reshape(img, [64, 64, 3])
    
    return img

def _parse_example(example, data_shape):
    tensor = tf.io.parse_single_example(example, data_shape)
    
    datapoint = {
        'anchor': decode_image(tensor['anchor']),
        'pos': decode_image(tensor['pos']),
        'neg': decode_image(tensor['neg'])
    }
    
    return datapoint

def load_data(tfrecord_path, batch_size=128):
    dataset = tf.data.TFRecordDataset(tfrecord_path)

    data_shape = {
    'anchor': tf.io.FixedLenFeature([], tf.string),
    'pos': tf.io.FixedLenFeature([], tf.string),
    'neg': tf.io.FixedLenFeature([], tf.string)
    }

    parser = partial(_parse_example, data_shape=data_shape)
    dataset = dataset.map(parser).shuffle(2048).batch(batch_size, drop_remainder=False).prefetch(2)

    return dataset