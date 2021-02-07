import matplotlib.pyplot as plt
import numpy as np
import os
import tensorflow as tf
from tensorflow import Tensor
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from keras.callbacks import *
import warnings
warnings.filterwarnings('ignore')
AUTOTUNE = tf.data.experimental.AUTOTUNE

# Tensorflow GPU configuration
print(tf.__version__)
print("Num GPUs Available: ", len(tf.config.experimental.list_physical_devices('GPU')))
print(os.getenv("CUDA_VISIBLE_DEVICES"))
tf.config.set_soft_device_placement(True)
def get_n_cores():
  nslots = os.getenv('NSLOTS')
  if nslots is not None:
    return int(nslots)
  raise ValueError('Environment variable NSLOTS is not defined.')
print("NUM CORES: ", get_n_cores())
tf.config.threading.set_intra_op_parallelism_threads(get_n_cores()-1)
tf.config.threading.set_inter_op_parallelism_threads(1)


# Roots folders
#main_root = '/projectnb2/cs542-bap/class_challenge/'
#main_root = "C:/Users/Juan Manuel/Desktop/Challenge_ML/"
data_folder_path = 'data/'
images_folder_path = '/projectnb2/cs542-bap/class_challenge/images/'

# Files
classes_path =  data_folder_path+'classes.txt'
train_path =  data_folder_path+'train.txt'
val_path =  data_folder_path+'val.txt' 
test_path =  data_folder_path+'test.txt'
with open(classes_path, 'r') as f:
    all_classes = [int(c.strip()) for c in f.readlines()]
class_all = list([str(x) for x in all_classes if x in range(0, 1010)])

# Hypeparameters
model_name = "R18-global-0-dense"
batch_size = 32
lr = 0.0001
ep = 50
drop = 0.4


#-----------------------------------------------------------------------------
# Prepare features
def configure_for_performance(ds, batch_size):
    ds = ds.cache()
    ds = ds.shuffle(buffer_size=1000)
    ds = ds.batch(batch_size)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

@tf.autograph.experimental.do_not_convert
def prepare_datasets(file_path, flag_label_in_ds, class_names, train_flag, batch_size):
    
    # Retrieve image list from file
    ds = tf.data.TextLineDataset(file_path)
    dataset_length = [i for i,_ in enumerate(ds)][-1] + 1
    print("Dataset Lenght ",dataset_length)
    for f in ds.take(1):
         print("Example content element dataset: ",f.numpy())
         
     # Encode image 
    def decode_img_train(img, crop_size=224):
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        return tf.image.resize(img, [crop_size, crop_size])
    def get_label(label):
        one_hot = tf.where(tf.equal(label, class_names))
        return tf.reduce_min(one_hot)
    def process_path_label(file_path):
        file_path = tf.strings.split(file_path)
        label = get_label(file_path[1])
        img = decode_img_train(tf.strings.join([images_folder_path, file_path[0], '.jpg']))
        return img, label
    #for test
    def decode_img(img, crop_size=224):#224 155
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        return tf.image.resize(img, [crop_size, crop_size])
    def process_path(file_path):
        file_path = tf.strings.split(file_path)
        img = decode_img(tf.strings.join([images_folder_path, file_path[0], '.jpg']))
        return img, file_path
    def process_path_label_val(file_path):
        file_path = tf.strings.split(file_path)
        label = get_label(file_path[1])
        img = decode_img(tf.strings.join([images_folder_path, file_path[0], '.jpg']))
        return img, label
    if train_flag:
        ds = ds.map(process_path_label, num_parallel_calls=AUTOTUNE)
        for image,_ in ds.take(1):
            image_shape = image.numpy().shape
    elif flag_label_in_ds:
        ds = ds.map(process_path_label_val, num_parallel_calls=AUTOTUNE)
        for image,_ in ds.take(1):
            image_shape = image.numpy().shape
    else:
        ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
        for image, _ in ds.take(1):
            image_shape = image.numpy().shape
    
    ds = configure_for_performance(ds, batch_size)
        
    if flag_label_in_ds:
        # Check some images
        image_batch, label_batch = next(iter(ds))
        plt.figure(figsize=(10, 10))
        for i in range(8):
            plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(class_names[label_batch[i]])
            plt.axis("off")
    else:
        # Check some images
        image_batch, _ = next(iter(ds))
        plt.figure(figsize=(10, 10))
        for i in range(8):
            plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            #plt.title(class_names[label_list[i]])
            plt.axis("off")

    return ds, image_shape

#-----------------------------------------------------------------------------
# Build model
def get_resnet50(image_shape):
    from tensorflow.keras.applications.resnet import ResNet50
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_shape))
    for layer in model.layers:
        layer.trainable = False
    model.summary()
    return model
def get_model_resnet50(image_shape, class_out, weights, drop):
    model = tf.keras.Sequential([
        layers.experimental.preprocessing.RandomFlip(mode='horizontal'),
        layers.experimental.preprocessing.RandomZoom(0.2),
        layers.experimental.preprocessing.RandomTranslation(0.2, 0.2),
        get_resnet50(image_shape, weights),
        GlobalAveragePooling2D(),
        Dropout(drop),
        Dense(class_out)
    ])
    return model
# Custom Resnet
def relu_bn(inputs: Tensor) -> Tensor:
    relu = ReLU()(inputs)
    bn = BatchNormalization()(relu)
    return bn
def residual_block(x: Tensor, downsample: bool, filters: int, kernel_size: int = 3) -> Tensor:
    y = Conv2D(kernel_size=kernel_size,strides= (1 if not downsample else 2), filters=filters,padding="same")(x)
    y = relu_bn(y)
    y = Conv2D(kernel_size=kernel_size,strides=1,filters=filters,padding="same")(y)
    if downsample:
        x = Conv2D(kernel_size=1,strides=2,filters=filters,padding="same")(x)
    out = Add()([x, y])
    out = relu_bn(out)
    return out
def get_model_resnet34(image_shape, num_classes, drop):    
    inputs = Input(shape=(image_shape)) 
    data_augmentation = keras.Sequential([
        layers.experimental.preprocessing.RandomZoom(0.2),
        layers.experimental.preprocessing.RandomRotation(0.2),
        layers.experimental.preprocessing.RandomTranslation(0.2, 0.2),
        layers.experimental.preprocessing.Rescaling(1./255)
    ], name='data_augmentation')
    x = data_augmentation(inputs)
    x = Conv2D(kernel_size=7,strides=2,filters=64,padding="same")(x)
    x = BatchNormalization()(x)
    x = relu_bn(x)
    x = MaxPool2D(pool_size=(3, 3), strides=2, padding="same")(x)    
    num_blocks_list = [2,2,2,2]#[2, 2, 2, 2]#[3, 5, 6, 3]
    num_filters_list = [64, 128, 256, 512]    
    for i in range(len(num_blocks_list)):
        num_blocks = num_blocks_list[i]
        num_filters = num_filters_list[i]
        for j in range(num_blocks):
            x = residual_block(x, downsample=(j==0 and i!=0), filters=num_filters)    
    x = GlobalAveragePooling2D()(x)
    x = Dense(1024, activation='relu')(x)
    outputs = Dense(num_classes)(x)  
    model = Model(inputs, outputs)
    return model


#-----------------------------------------------------------------------------
# Train
def sl_training(model, train_ds, val_ds, ep, lr, save_model):
    
    
    # Create Datasets
    num_classes = len(class_names)
    print("--Train dataset")
    train_ds, _ = pf.prepare_datasets(file_train, True, class_names, True, batch_size)
    print("--Val dataset")
    val_ds, image_shape = pf.prepare_datasets(file_val, True, class_names, False, batch_size)
    print("------>Num total classes: ",num_classes)
    print("------>Image shape: ", image_shape)
    
    print("----------------------------- NAME: ", model_name," - BATCH: ", str(batch_size), " - LR: ",str(lr))
    # Save paths
    score_path = aod.results_path+model_name+"-Final-SL-model.txt"
    save_history_path = aod.results_path+model_name+"-history-batch-"+str(batch_size)+"-LR-"+str(lr)+".png"
    save_model = aod.models_saved_path+model_name+"GO-lr"+str(lr)+"-batch"+str(batch_size)
    
    # Build model 
    model = m_s.get_model_resnet34(image_shape, num_classes, drop)
    #model_load_path = 'models_saved/'+'R18-Flatten-GO-lr0.0001-batch32-15-0.23.h5'"'
    #model.load_weights(model_load_path)
    model.summary()
    
    #Hyperparam
    top_k = 5
    
    #Train model
    es = keras.callbacks.EarlyStopping(monitor='val_accuracy', mode='auto', verbose=1, patience=5)
    checkpoint = ModelCheckpoint(model_save_checkp, save_weights_only=True, verbose=1, monitor='val_accuracy', save_best_only=True, mode='auto') 
    filepath = save_model+"-{epoch:02d}-{val_accuracy:.2f}.h5"
    checkpoint = ModelCheckpoint(filepath, monitor='val_loss', verbose=2, save_best_only=False, save_weights_only=True, mode='auto')
    learning_rate_reduction = keras.callbacks.ReduceLROnPlateau(monitor='val_loss', patience=3, verbose=1, factor=0.5, min_lr=0.000001, cooldown=1)
    opt = keras.optimizers.Adam(lr=lr)
    #opt = tf.keras.optimizers.SGD(learning_rate=0.001, momentum=0.9)
    model.compile(optimizer=opt,
                   loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                   metrics=['accuracy',tf.keras.metrics.SparseTopKCategoricalAccuracy(k=top_k)])
    history = model.fit(train_ds, validation_data = val_ds, epochs=ep, verbose=1, shuffle=True, callbacks=[checkpoint, learning_rate_reduction])
    


#-----------------------------------------------------------------------------
# Launch model
if __name__ == '__main__':
    
    sl_training(model, train_ds, val_ds, ep, lr, save_model)

    # Get challenge scores
    '''print("Get challenge scores")
    print("--Test dataset")
    test_ds,image_shape = pf.prepare_datasets(file_test, False, class_names, False, 32)
    print("model load")
    model_load_path = aod.models_saved_path+"R18-res-2.0-cont-lr0.0001-batch16-15-0.38.h5"
    model_load = m_s.get_model_resnet34(image_shape, num_classes, 0)
    model_load.load_weights(model_load_path)
    model_load.compile(optimizer=keras.optimizers.Adam(),
                       loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True),
                       metrics=['accuracy',tf.keras.metrics.SparseTopKCategoricalAccuracy(k=5)])
          
    with open('submission_task1_supervised.csv', 'w') as f:
        f.write('id,predicted\n')
        j=0
        #dataset_length = [i for i,_ in enumerate(test_ds)][-1] + 1
        #print("Dataset Lenght ",dataset_length)
        for image_batch, image_names in test_ds:
            if j%100==0:
                print("round=",str(j),"/",)
            for image_name, predictions in zip(image_names.numpy(), model_load.predict(image_batch)):
                inds = np.argpartition(predictions, -5)[-5:]
                line = str(int(image_name)) + ',' + ' '.join([class_names[i] for i in inds])
                f.write(line + '\n')
            j+=1'''