import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

class ChannelAttention(layers.Layer):
    def __init__(self, in_planes, ratio=4, **kwargs):
        super().__init__(**kwargs)

        self.avg_pool = layers.GlobalAveragePooling2D(keepdims=True)
        self.max_pool = layers.GlobalMaxPooling2D(keepdims=True)

        self.fc = keras.models.Sequential([
            layers.Conv2D(in_planes // ratio + 1, 1, use_bias=False, padding='same'),
            layers.ReLU(),
            layers.Conv2D(in_planes, 1, use_bias=False, padding='same')
        ])

    def call(self, x):
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))

        out = avg_out + max_out
        out = keras.activations.sigmoid(out)

        return out


class SpatialAttention(layers.Layer):
    def __init__(self, kernel_size=7, **kwargs):
        super().__init__(**kwargs)

        self.conv = layers.Conv2D(1, kernel_size, use_bias=False, padding='same')

    def call(self, x):
        avg_out = tf.reduce_mean(x, axis=-1, keepdims=True)
        max_out = tf.reduce_max(x, axis=-1, keepdims=True)
        
        x = tf.concat([avg_out, max_out], axis=-1)
        x = self.conv(x)
        out = keras.activations.sigmoid(x)

        return out

class CBAM(layers.Layer):
    def __init__(self, in_planes, ratio=4, kernel_size=7, **kwargs):
        super().__init__(**kwargs)

        self.channel_attention = ChannelAttention(in_planes, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def call(self, x):
        x = self.channel_attention(x) * x
        x = self.spatial_attention(x) * x

        return x