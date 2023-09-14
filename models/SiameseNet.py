import time

import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from keras_vggface.vggface import VGGFace

from models.DistanceLayer import DistanceLayer

def build_network(base_network):
    anchor_input = layers.Input(name="anchor", shape=[64, 64, 3])
    positive_input = layers.Input(name="pos", shape=[64, 64, 3])
    negative_input = layers.Input(name="neg", shape=[64, 64, 3])

    distances = DistanceLayer()(
        base_network(anchor_input),
        base_network(positive_input),
        base_network(negative_input),
    )

    siamese_network = keras.Model(
        inputs=[anchor_input, positive_input, negative_input], outputs=distances
    )

    return siamese_network

class SiameseModel(keras.Model):
    """The Siamese Network model with a custom training and testing loops.

    Computes the triplet loss using the three embeddings produced by the
    Siamese Network.

    The triplet loss is defined as:
       L(A, P, N) = max(‖f(A) - f(P)‖² - ‖f(A) - f(N)‖² + margin, 0)
    """

    def __init__(self, siamese_network, margin=0.5):
        super().__init__()
        self.siamese_network = siamese_network
        self.margin = margin

        self.loss_tracker = keras.metrics.Mean(name="loss")
        self.val_loss_tracker = keras.metrics.Mean(name="loss")

        self.accuracy_tracker = keras.metrics.Accuracy(name='accuracy')
        self.val_accuracy_tracker = keras.metrics.Accuracy(name='accuracy')

        self.ap_mean_tacker = keras.metrics.Mean(name='ap_mean')
        self.ap_std_tacker = keras.metrics.Mean(name='ap_std')
        self.an_mean_tacker = keras.metrics.Mean(name='an_mean')
        self.an_std_tacker = keras.metrics.Mean(name='an_std')

        self.val_ap_mean_tacker = keras.metrics.Mean(name='ap_mean')
        self.val_ap_std_tacker = keras.metrics.Mean(name='ap_std')
        self.val_an_mean_tacker = keras.metrics.Mean(name='an_mean')
        self.val_an_std_tacker = keras.metrics.Mean(name='an_std')

    def call(self, inputs):
        return self.siamese_network(inputs)

    def train_step(self, data):
        # GradientTape is a context manager that records every operation that
        # you do inside. We are using it here to compute the loss so we can get
        # the gradients and apply them using the optimizer specified in
        # `compile()`.
        # print('Here :)')
        with tf.GradientTape() as tape:
            loss, ap, an = self._compute_loss(data)

        # Storing the gradients of the loss function with respect to the
        # weights/parameters.
        gradients = tape.gradient(loss, self.siamese_network.trainable_weights)

        # Applying the gradients on the model using the specified optimizer
        self.optimizer.apply_gradients(
            zip(gradients, self.siamese_network.trainable_weights)
        )

        # Let's update and return the training loss metric.
        self.loss_tracker.update_state(loss)

        self._computer_accuracy(ap, an)
        return {"loss": self.loss_tracker.result(),
                'accuracy': self.accuracy_tracker.result(),
                'ap_mean': self.ap_mean_tacker.result(),
                'ap_std': self.ap_std_tacker.result(),
                'an_mean': self.an_mean_tacker.result(),
                'an_std': self.an_std_tacker.result(),
                }

    def test_step(self, data):
        # self.siamese_network.save(f'{time.time()}.h5')
        loss, ap, an = self._compute_val_loss(data)
        # self._computer_accuracy(ap, an)
        self._computer_val_accuracy(ap, an)
        # Let's update and return the loss metric.
        self.val_loss_tracker.update_state(loss)
        return {"loss": self.val_loss_tracker.result(),
                'accuracy': self.val_accuracy_tracker.result(),
                'ap_mean': self.val_ap_mean_tacker.result(),
                'ap_std': self.val_ap_std_tacker.result(),
                'an_mean': self.val_an_mean_tacker.result(),
                'an_std': self.val_an_std_tacker.result(),
                }

    def _compute_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)
        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        
        return loss, ap_distance, an_distance

    def _compute_val_loss(self, data):
        # The output of the network is a tuple containing the distances
        # between the anchor and the positive example, and the anchor and
        # the negative example.
        ap_distance, an_distance = self.siamese_network(data)
        # Computing the Triplet Loss by subtracting both distances and
        # making sure we don't get a negative value.
        loss = ap_distance - an_distance
        loss = tf.maximum(loss + self.margin, 0.0)
        
        return loss, ap_distance, an_distance

    def _computer_accuracy(self, ap, an):
        ap_mean = tf.math.reduce_mean(ap)
        ap_std = tf.math.reduce_std(an)

        threshold = ap_mean + ap_std

        pos_mask = ap <= threshold
        neg_mask = an > threshold

        self.accuracy_tracker.update_state(pos_mask, tf.ones_like(pos_mask))
        self.accuracy_tracker.update_state(neg_mask, tf.zeros_like(neg_mask))

        self.ap_mean_tacker.update_state(ap_mean)
        self.ap_std_tacker.update_state(ap_std)
        self.an_mean_tacker.update_state(tf.math.reduce_mean(an))
        self.an_std_tacker.update_state(tf.math.reduce_std(an))

    def _computer_val_accuracy(self, ap, an):
        ap_mean = tf.math.reduce_mean(ap)
        ap_std = tf.math.reduce_std(an)

        threshold = ap_mean + ap_std

        pos_mask = ap <= threshold
        neg_mask = an > threshold

        self.val_accuracy_tracker.update_state(pos_mask, tf.ones_like(pos_mask))
        self.val_accuracy_tracker.update_state(neg_mask, tf.zeros_like(neg_mask))

        self.val_ap_mean_tacker.update_state(ap_mean)
        self.val_ap_std_tacker.update_state(ap_std)
        self.val_an_mean_tacker.update_state(tf.math.reduce_mean(an))
        self.val_an_std_tacker.update_state(tf.math.reduce_std(an))



    @property
    def metrics(self):
        # We need to list our metrics here so the `reset_states()` can be
        # called automatically.
        return [self.loss_tracker,
                self.accuracy_tracker, 
                self.ap_mean_tacker, 
                self.ap_std_tacker,
                self.an_mean_tacker, 
                self.an_std_tacker,
                
                self.val_loss_tracker,
                self.val_accuracy_tracker, 
                self.val_ap_mean_tacker, 
                self.val_ap_std_tacker,
                self.val_an_mean_tacker, 
                self.val_an_std_tacker,]