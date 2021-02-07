import matplotlib.pyplot as plt
import warnings
import numpy as np
from math import sqrt
import os
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import *
from tensorflow.keras.layers import *
from tensorflow.keras.models import *
from keras.callbacks import *
import matplotlib.image as mpimg
from sklearn import decomposition
import classification_models.tfkeras
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
#data_folder_path = '/projectnb2/cs542-bap/class_challenge/'
#images_folder_path = '/projectnb2/cs542-bap/class_challenge/images/'
data_folder_path = 'data/'
images_folder_path = 'C:/Users/Juan Manuel/Desktop/Challenge_ML/data/images/'

# Files
classes_ssl_path =  data_folder_path+'classes_held_out.txt'
train_u_path =  data_folder_path+'train_held_out.txt'   
train_l_path =  data_folder_path+'train_held_out_labeled.txt' 
val_ssl_path =  data_folder_path+'val_held_out.txt' 
test_ssl_path =  data_folder_path+'test_held_out.txt'
with open(classes_ssl_path, 'r') as f:
    classes_ssl = [int(c.strip()) for c in f.readlines()]
classes_ssl = list([str(x) for x in classes_ssl if x in range(0, 1010)])


#-----------------------------------------------------------------------------
# Prepare datasets
@tf.autograph.experimental.do_not_convert
def prepare_datasets_ssl(file_path, flag_label_in_ds, class_names, flag_train):
    
    # Retrieve image list from file
    ds = tf.data.TextLineDataset(file_path)
    dataset_length = [i for i,_ in enumerate(ds)][-1] + 1
    print("Dataset Lenght ",dataset_length)
    for f in ds.take(1):
         print("Example content element dataset: ",f.numpy())
         
    # Encode image 
    def decode_img_aug(img, crop_size=224):
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        img = tf.image.random_saturation(img, 0.8, 1.2)
        img = tf.image.random_contrast(img, 0.8, 1.2)
        return tf.image.resize(img, [crop_size, crop_size])
    def decode_img(img, crop_size=224):
        img = tf.io.read_file(img)
        img = tf.image.decode_jpeg(img, channels=3)
        return tf.image.resize(img, [crop_size, crop_size])
    def get_label(label):
        one_hot = tf.where(tf.equal(label, class_names))
        return tf.reduce_min(one_hot)
    def process_path_label(path):
        path = tf.strings.split(path)
        label = get_label(path[1])
        if flag_train:
            img = decode_img_aug(tf.strings.join([images_folder_path, path[0], '.jpg']))
        else:
            img = decode_img(tf.strings.join([images_folder_path, path[0], '.jpg']))
        return img, label
    def process_path(path):
        path = tf.strings.split(path)
        img = decode_img(tf.strings.join([images_folder_path, path[0], '.jpg']))
        return img
    if flag_label_in_ds:
        ds = ds.map(process_path_label, num_parallel_calls=AUTOTUNE)
        for image, _ in ds.take(1):
            image_shape = image.numpy().shape
    else:
        ds = ds.map(process_path, num_parallel_calls=AUTOTUNE)
        for image in ds.take(1):
            image_shape = image.numpy().shape
    
    # Create batches
    def configure_for_performance(ds, batch_size):
        #ds = ds.cache()
        #ds = ds.shuffle(buffer_size=1000)
        ds = ds.batch(batch_size)
        ds = ds.prefetch(buffer_size=AUTOTUNE)
        return ds
    batch_size = 4
    ds = configure_for_performance(ds, batch_size)
    
    # Get other infos from files
    def get_file_content(path):
        with open(path, 'r') as f:
            content = [c.strip() for c in f.readlines()]
        return content
    def get_file_imid_label(path):
        with open(path) as f:
            content = f.readlines()
            img = [x.split(" ")[0].strip() for x in content]
            label = [x.split(" ")[1].strip() for x in content]
        return img, label
    label_list = []
    image_list = []
    if flag_label_in_ds:
        # Get labels for training (for PCA)
        image_list, label_list = get_file_imid_label(file_path)
        label_list = np.array([class_names.index(el) for el in label_list])
        print("Label list size: ",label_list.size)
        print("Label list example: ",label_list[0])
    else:
        # Fake labels (for PCA)
        image_list = get_file_content(file_path)
        label_list = np.array(["?" for el in image_list])
        print("Label list size: ",label_list.size)
        print("Label list example: ",label_list[0])
        
    '''if flag_label_in_ds:
        # Check some images
        image_batch, label_batch = next(iter(ds))
        plt.figure(figsize=(10, 10))
        for i in range(2):
            plt.subplot(3, 3, i + 1)
            plt.imshow(image_batch[i].numpy().astype("uint8"))
            plt.title(class_names[label_list[i]]+" "+class_names[label_batch[i]])
            plt.axis("off")'''

    return ds, label_list, image_list, image_shape

#-----------------------------------------------------------------------------
# Visualize data and stats
def plot_PCA(features, labels, class_names, save_path):
    n_train, x, y, z = features.shape
    numFeatures = x * y * z
    pca = decomposition.PCA(n_components = 2)
    X = features.reshape((n_train, x*y*z))
    pca.fit(X)
    C = pca.transform(X)
    C1 = C[:,0]
    C2 = C[:,1]
    plt.subplots(figsize=(10,10))
    for i, class_name in enumerate(class_names):
        plt.scatter(C1[labels == i][:1000], C2[labels == i][:1000], label = class_name, alpha=0.4)
    plt.legend()
    plt.title("PCA Projection")
    plt.savefig(save_path)
    plt.show()

def plot_PCA_no_l(features, labels, class_names, save_path):
    n_train, x, y, z = features.shape
    numFeatures = x * y * z
    pca = decomposition.PCA(n_components = 2)
    X = features.reshape((n_train, x*y*z))
    pca.fit(X)
    C = pca.transform(X)
    C1 = C[:,0]
    C2 = C[:,1]
    plt.subplots(figsize=(10,10))
    #for i, class_name in enumerate(class_names):
    plt.scatter(C1, C2, label = "?", alpha=0.4)
    plt.legend()
    plt.title("PCA Projection")
    plt.savefig(save_path)
    plt.show()

def plot_accuracy_loss(history, flag_val, save_history_path):
    #['accuracy', 'loss', 'val_accuracy', 'val_loss']
    fig = plt.figure(figsize=(10,5))
    # Plot accuracy
    plt.subplot(221)
    plt.plot(history.history['accuracy'],'bo--', label = "acc")
    if flag_val:
        plt.plot(history.history['val_accuracy'], 'ro--', label = "val_acc")
    plt.title("train_acc vs val_acc")
    plt.ylabel("accuracy")
    plt.xlabel("epochs")
    plt.legend()

    # Plot loss function
    plt.subplot(222)
    plt.plot(history.history['loss'],'bo--', label = "loss")
    if flag_val:
        plt.plot(history.history['val_loss'], 'ro--', label = "val_loss")
    plt.title("train_loss vs val_loss")
    plt.ylabel("loss")
    plt.xlabel("epochs")

    plt.legend()
    plt.savefig(save_history_path)
    plt.show()

#-----------------------------------------------------------------------------
# Models
def get_resnet18(image_shape):
    from classification_models.tfkeras import Classifiers
    Resnet18, process_input = Classifiers.get('resnet18')
    model = Resnet18(image_shape, weights='imagenet', include_top=False)
    return model
def get_resnet34(image_shape):
    from classification_models.tfkeras import Classifiers
    Resnet34, process_input = Classifiers.get('resnet34')
    model = Resnet34(image_shape, weights='imagenet', include_top=False)
    return model
def get_resnet50(image_shape):
    from tensorflow.keras.applications.resnet import ResNet50
    model = ResNet50(weights='imagenet', include_top=False, input_shape=(image_shape))
    return model
def get_resnet101(image_shape):
    from tensorflow.keras.applications.resnet import ResNet101
    model = ResNet101(weights='imagenet', include_top=False, input_shape=(image_shape))
    return model
def get_vgg16(image_shape):
    from tensorflow.keras.applications.vgg16 import VGG16
    model = VGG16(weights='imagenet', include_top=False, input_shape=(image_shape))
    return model
def get_vgg19(image_shape):
    from tensorflow.keras.applications.vgg19 import VGG19
    model = VGG19(weights='imagenet', include_top=False, input_shape=(image_shape))
    return model
def get_densenet(image_shape):
    from tensorflow.keras.applications.densenet import DenseNet121
    model = DenseNet121(weights='imagenet', include_top=False, input_shape=(image_shape))
    return model
def get_mobilenet(image_shape):
    from tensorflow.keras.applications.mobilenet import MobileNet
    model = MobileNet(weights='imagenet', include_top=False, input_shape=(image_shape))
    return model
def get_xception(image_shape):
    from tensorflow.keras.applications.xception import Xception
    model = Xception(weights='imagenet', include_top=False, input_shape=(image_shape))
    return model

def get_model_head(feature_shape, class_out):
    model = tf.keras.Sequential([
        GlobalAveragePooling2D(),
        Dense(512, activation="relu"),
        Dropout(0.4),
        Dense(class_out, activation="sigmoid")
    ])
    '''
    inputs = Input(shape=(feature_shape))
    x = GlobalAveragePooling2D()(inputs)
    x = Dense(128, activation="relu")(x)
    x = Dropout(0.4)(x)
    outputs = Dense(class_out, activation="sigmoid")(x) 
    model = Model(inputs, outputs)
    '''
    model.compile(optimizer=keras.optimizers.Adam(lr=0.0001), 
                  loss=tf.losses.SparseCategoricalCrossentropy(from_logits=True), 
                  metrics=['accuracy'])
    return model

#-----------------------------------------------------------------------------
# Train ssl model
def ssl_training(train_l_path, train_u_path, val_ssl_path, test_ssl_path, class_names):
    
    # Retrieve data
    print("---Prepare dataset train L")
    train_l_ds, train_l_label_list, train_l_image_list, image_shape = prepare_datasets_ssl(train_l_path, True, class_names, False)
    print("---Prepare dataset train U")
    train_u_ds, train_u_fake_label_list, train_u_image_list, _ = prepare_datasets_ssl(train_u_path, False, class_names, False)
    print("---Prepare dataset val")
    val_l_ds, val_l_label_list, val_l_image_list, _ = prepare_datasets_ssl(val_ssl_path, True, class_names, True)
    print("--Classes")
    print(class_names)
    print("num clases: ",len(class_names))
    
    # Backbone model Resnet
    base_model = get_resnet34(image_shape)
    train_l_features = base_model.predict(train_l_ds)
    train_u_features = base_model.predict(train_u_ds)
    val_l_features = base_model.predict(val_l_ds)
    n_train, x, y, z = train_l_features.shape
    
    # Visualize data features out of Resnet in lower dimension
    '''
    plot_PCA(train_l_features, train_l_label_list, class_names, "pca-x-SSL-train-l.png")
    plot_PCA_no_l(train_u_features, train_u_fake_label_list, class_names, "pca-SSL-train-u.png")
    plot_PCA(val_l_features, val_l_label_list, class_names, "pca-SSL-val.png")
    '''
    
    # Head model first round
    ep = 15
    model = get_model_head((x, y, z), len(class_names))
    history = model.fit(train_l_features, train_l_label_list, validation_data = (val_l_features,val_l_label_list), batch_size=4, epochs=ep, verbose=2, shuffle=True)
    scores = model.evaluate(val_l_features,val_l_label_list)
    print("Scores: ", scores[1])         
    plot_accuracy_loss(history, True, "SSL-history.png")
    
    
    # Self training algorithm
    ep = 15
    train_l_label_list = np.array(train_l_label_list)
    max_gen_class = np.zeros(20)
    while len(train_l_features) < 1000:
        print("-----U ",len(train_u_features))
        print("-----L ", len(train_l_features))
        
        # Predict U
        prediction = model.predict(train_u_features)
        
        # Move label predicted
        classAdded = []
        maxInRows = np.amax(prediction, axis=1)
        index_top = maxInRows.argsort()[:][::-1]
        index_top = np.sort(index_top)
        index_top = index_top[::-1]
        max_gen = np.argmax(max_gen_class)
        for ind in index_top:
            maxValue = maxInRows[ind]
            feature_max = train_u_features[ind]
            pred_class_max = np.argmax(prediction[ind])
            if len(classAdded) >= 20:
                break
            elif maxValue < 0.95 or pred_class_max in classAdded:
                continue # skip if sample not in the rules
            else:
                # Extract features and label
                print("max Value: ", maxValue)
                print("pred_class_max: ", pred_class_max)
                # Add row to L and delete form U
                train_l_features = np.insert(train_l_features, -1, np.array([feature_max]), axis=0)
                train_l_label_list = np.insert(train_l_label_list, -1, np.array([pred_class_max]), axis=0)
                train_u_features = np.delete(train_u_features, (ind), axis=0)
                # Keep track of the added classes
                classAdded.append(pred_class_max)
                print("Class added: ",classAdded)
         
        model = get_model_head((x, y, z), len(class_names))    
        history = model.fit(train_l_features,train_l_label_list, validation_data = (val_l_features,val_l_label_list), epochs=ep, batch_size=4, verbose=0, shuffle=True)
        scores = model.evaluate(val_l_features,val_l_label_list)
        print("Scores: ", scores[1])          
        plot_accuracy_loss(history, True, "SSL-history.png")
        #model.save_weights('modelSSL-'+str(scores[1])+'.h5')
        
            
    print("----------------------------- GET CHALLENGE SCORES")
    test_u_ds, _, test_image_list, _ = prepare_datasets_ssl(test_ssl_path, False, class_names, False)
    base_model = get_resnet34(image_shape, 'imagenet')
    test_features = base_model.predict(test_u_ds)
    _, x, y, z = test_features.shape
    model = get_model_head((x, y, z), len(class_names))
    #model.load_weights('modelSSL-0.7194719314575195.h5')
    test_features = base_model.predict(test_u_ds)
    prediction = model.predict(test_features)
    with open('submission_task2_supervised.csv', 'w') as f:
        f.write('id,predicted\n')
        for i in range(len(test_image_list)):
            ind = np.argmax(prediction[i])
            line = str(test_image_list[i]) + ','+class_names[ind]
            f.write(line + '\n')
           
            
#-----------------------------------------------------------------------------
# Launch model
if __name__ == '__main__':
    
    ssl_training(train_l_path, train_u_path, val_ssl_path, test_ssl_path, classes_ssl)
    
    
    
    
    
    
    
    
    
    