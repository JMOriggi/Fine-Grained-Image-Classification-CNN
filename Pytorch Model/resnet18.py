# Import TensorFlow and other libraries
import pathlib
import numpy as np
import os
import tensorflow as tf

from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.models import Sequential

import time

# Explore the dataset
data_dir = '/projectnb2/cs542-bap/class_challenge/'
image_dir = os.path.join(data_dir, 'images')
image_dir = pathlib.Path(image_dir)

train_ds = tf.data.TextLineDataset(os.path.join(data_dir, 'train.txt'))
val_ds = tf.data.TextLineDataset(os.path.join(data_dir, 'val.txt'))
test_ds = tf.data.TextLineDataset(os.path.join(data_dir, 'test.txt'))

with open(os.path.join(data_dir, 'classes.txt'), 'r') as f:
    class_names = [c.strip() for c in f.readlines()]

num_classes = len(class_names)


# Write a short function that converts a file path to an (img, label) pair:
def decode_img(img, crop_size=224):
    img = tf.io.read_file(img)
    # convert the compressed string to a 3D uint8 tensor
    img = tf.image.decode_jpeg(img, channels=3)
    # resize the image to the desired size
    return tf.image.resize(img, [crop_size, crop_size])


def get_label(label):
    # find teh matching label
    one_hot = tf.where(tf.equal(label, class_names))
    # Integer encode the label
    return tf.reduce_min(one_hot)


def process_path(file_path):
    # should have two parts
    file_path = tf.strings.split(file_path)
    # second part has the class index
    label = get_label(file_path[1])
    # load the raw data from the file
    img = decode_img(tf.strings.join(
        [data_dir, 'images/', file_path[0], '.jpg']))
    return img, label


def process_path_test(file_path):
    # load the raw data from the file
    img = decode_img(tf.strings.join([data_dir, 'images/', file_path, '.jpg']))
    return img, file_path


# Finish setting up data
batch_size = 32

# Set `num_parallel_calls` so multiple images are loaded/processed in parallel.
AUTOTUNE = tf.data.experimental.AUTOTUNE
train_ds = train_ds.map(process_path, num_parallel_calls=AUTOTUNE)
val_ds = val_ds.map(process_path, num_parallel_calls=AUTOTUNE)
test_ds = test_ds.map(process_path_test, num_parallel_calls=AUTOTUNE)

for image, label in train_ds.take(1):
    print("Image shape: ", image.numpy().shape)
    print("Label: ", label.numpy())


# Data loader hyper-parameters for performance!


def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds


train_ds = configure_for_performance(train_ds)
val_ds = configure_for_performance(val_ds)
test_ds = configure_for_performance(test_ds)


class BasicBlock(layers.Layer):
    """A single residual block"""

    def __init__(self, filter_num, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = layers.Conv2D(
            filters=filter_num,
            kernel_size=(3, 3),
            strides=stride,
            padding="same")
        self.bn1 = layers.BatchNormalization()
        self.conv2 = layers.Conv2D(
            filters=filter_num,
            kernel_size=(3, 3),
            strides=1,
            padding="same")
        self.bn2 = layers.BatchNormalization()
        if stride != 1:
            self.downsample = Sequential()
            self.downsample.add(
                layers.Conv2D(
                    filters=filter_num,
                    kernel_size=(1, 1),
                    strides=stride))
            self.downsample.add(layers.BatchNormalization())
        else:
            self.downsample = lambda x: x

    def call(self, inputs, training=None, **kwargs):
        residual = self.downsample(inputs)

        x = self.bn1(inputs, training=training)
        x = tf.nn.relu(x)
        x = self.conv1(x)

        x = self.bn2(x, training=training)
        x = tf.nn.relu(x)
        x = self.conv2(x)

        output = layers.add([residual, x])

        return output


def make_basic_block_layer(filter_num, blocks, stride=1):
    """A stack of residual blocks"""
    res_block = Sequential()
    res_block.add(BasicBlock(filter_num, stride=stride))

    for _ in range(1, blocks):
        res_block.add(BasicBlock(filter_num, stride=1))

    return res_block


class ResNetTypeI(tf.keras.Model):
    """ResNet"""

    def __init__(self, layer_params):
        super(ResNetTypeI, self).__init__()
        self.conv1 = layers.Conv2D(
            filters=64,
            kernel_size=(7, 7),
            strides=2,
            padding="same")
        self.bn1 = layers.BatchNormalization()
        self.pool1 = layers.MaxPool2D(
            pool_size=(3, 3),
            strides=2,
            padding="same")

        self.layer1 = make_basic_block_layer(
            filter_num=64,
            blocks=layer_params[0])
        self.layer2 = make_basic_block_layer(
            filter_num=128,
            blocks=layer_params[1],
            stride=2)
        self.layer3 = make_basic_block_layer(
            filter_num=256,
            blocks=layer_params[2],
            stride=2)
        self.layer4 = make_basic_block_layer(
            filter_num=512,
            blocks=layer_params[3],
            stride=2)

        self.pool2 = layers.GlobalAveragePooling2D()
        self.fc = layers.Dense(units=num_classes)

    def call(self, inputs, training=None, mask=None):
        x = self.conv1(inputs)
        x = self.bn1(x, training=training)
        x = tf.nn.relu(x)
        x = self.pool1(x)
        x = self.layer1(x, training=training)
        x = self.layer2(x, training=training)
        x = self.layer3(x, training=training)
        x = self.layer4(x, training=training)
        x = self.pool2(x)
        output = self.fc(x)

        return output


# ResNet18

model = Sequential([
    layers.experimental.preprocessing.RandomFlip(
        mode='horizontal'),
    layers.experimental.preprocessing.RandomZoom(0.2),
    layers.experimental.preprocessing.RandomTranslation(0.2, 0.2),
    layers.experimental.preprocessing.Rescaling(1./255),
    ResNetTypeI(layer_params=[2, 2, 2, 2])
])

# The usual loss function
learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(
    monitor='val_loss', patience=2, verbose=1, factor=0.5)

model.compile(optimizer=keras.optimizers.Adam(learning_rate=0.0001),
              loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
              metrics=['accuracy',
                       tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])

# Timestamp
timestr = time.strftime("%Y%m%d-%H%M%S")

# Training
cp_callback = tf.keras.callbacks.ModelCheckpoint(
    filepath=os.path.join("checkpoints-resnet18",
                          timestr,
                          "cp-{epoch:04d}.ckpt"),
    verbose=1,
    save_weights_only=True,
    save_freq='epoch',
    period=10)

# Logs
tb_callback = tf.keras.callbacks.TensorBoard(
    log_dir=os.path.join("logs-resnet18", timestr))

model.fit(train_ds,
          validation_data=val_ds,
          epochs=50,
          shuffle=True,
          callbacks=[
              cp_callback,
              learning_rate_reduction,
              tb_callback
          ])

# Output submission csv for Kaggle
with open(timestr + ".csv", 'w') as f:
    f.write('id,predicted\n')
    f.flush()
    for image_batch, image_names in test_ds:
        predictions = model.predict(image_batch)
        for image_name, predictions in zip(image_names.numpy(),
                                           model.predict(image_batch)):
            inds = np.argpartition(predictions, -5)[-5:]
            line = str(int(image_name)) + ',' + \
                ' '.join([class_names[i] for i in inds])
            f.write(line + '\n')
        f.flush()
