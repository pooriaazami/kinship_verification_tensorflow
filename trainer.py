import tensorflow as tf
import tensorflow.keras as keras
import tensorflow.keras.layers as layers

from keras_vggface.vggface import VGGFace

from utils.DataLoader import load_data
from models.SiameseNet import SiameseModel, build_network
from models.Attention import CBAM

def main():
    train_dataset = load_data('data\\KinFaceWIITrainFolds.tfrecords')#.take(10)
    validation_dataset = load_data('data\\KinFaceWIITestFolds.tfrecords')#.take(1)

    vgg = VGGFace(model='vgg16', include_top=False, input_shape=(64, 64, 3))

    input_layer = layers.Input((64, 64, 3))
    cbam = CBAM(3)(input_layer)
    x = layers.Add()([cbam, input_layer])
    # x = CBAM(3)(input_layer)
    x = vgg(x)
    x = layers.Flatten()(x)
    # x = layers.Dense(4096, activation='relu')(x)
    output_layer = layers.Dense(4096)(x)

    network = keras.Model(inputs=input_layer, outputs=output_layer)

    for layer in vgg.layers:
        layer.trainable  = False

    network.summary()
    vgg.summary()
    # network = keras.models.Sequential([
    #     CBAM(3),

    #     layers.Conv2D(64, 3, padding='same', input_shape=[64, 64, 3], activation='relu'),
    #     layers.Conv2D(64, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(2),

    #     layers.Conv2D(128, 3, padding='same', activation='relu'),
    #     layers.Conv2D(128, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(2),

    #     layers.Conv2D(256, 3, padding='same', activation='relu'),
    #     layers.Conv2D(256, 3, padding='same', activation='relu'),
    #     layers.Conv2D(256, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(2),

    #     layers.Conv2D(512, 3, padding='same', activation='relu'),
    #     layers.Conv2D(512, 3, padding='same', activation='relu'),
    #     layers.Conv2D(512, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(2),

    #     layers.Conv2D(512, 3, padding='same', activation='relu'),
    #     layers.Conv2D(512, 3, padding='same', activation='relu'),
    #     layers.Conv2D(512, 3, padding='same', activation='relu'),
    #     layers.MaxPooling2D(2),

    #     layers.Flatten(),

    #     layers.Dense(4096, activation='relu'),
    #     layers.Dropout(.4),
    #     layers.Dense(4096, activation='relu'),
    # ])

    siamese_network = build_network(network)
    model = SiameseModel(siamese_network)
    # model.build([[None, 64, 64, 3], [None, 64, 64, 3], [None, 64, 64, 3]])
    model.compile(optimizer=keras.optimizers.Adam(0.0001))
    
    # checkpoint = tf.keras.callbacks.ModelCheckpoint(filepath='tmp/checkpoints/', monitor='val_accuracy')
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_accuracy')
    
    model.fit(train_dataset, epochs=10, validation_data=validation_dataset)


if __name__ == '__main__':
    main()