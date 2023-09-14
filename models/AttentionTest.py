import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from Attention import ChannelAttention, SpatialAttention, CBAM

import unittest

class ChannelAttentionTest(unittest.TestCase):
    def test_channel_attention(self):
        zeros = tf.zeros((10, 64, 64, 3))

        layer = ChannelAttention(3)
        output = layer(zeros)

        print('[log]: (channel): ', output.shape)

    def test_spatial_attention(self):
        zeros = tf.zeros((10, 64, 64, 3))

        layer = SpatialAttention()
        output = layer(zeros)

        print('[log]: (spatial): ', output.shape)

    def test_cbam(self):
        zeros = tf.zeros((10, 64, 64, 3))

        layer = CBAM(3)
        output = layer(zeros)

        print('[log]: (CBAM): ', output.shape)



if __name__ == '__main__':
    unittest.main()