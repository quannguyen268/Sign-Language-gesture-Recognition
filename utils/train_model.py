from tensorflow.keras import layers
from tensorflow.keras.models import  Model, load_model
from tensorflow.keras.applications import VGG16
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import preprocess_input
from sklearn.model_selection import train_test_split
from tensorflow.keras import backend as K
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from tensorflow.keras.callbacks import ReduceLROnPlateau
import random
import tqdm
import h5py
import numpy as np
import os




# Define customized model
class miniVGG:
    def build(height, width, depth, include_top=True, pooling='avg'):
        input_shape = (height, width, depth)
        chanDim = -1

        if K.image_data_format() == "channels_first":
            input_shape = (depth, height, width)
            chanDim = 1
        input = layers.Input(shape=(96,96,1), name='input')
        # Block 1
        x = layers.Conv2D(
            32, (3, 3), activation='swish', padding='same', name='block1_conv1')(input)
        x = layers.BatchNormalization()(x)
        x = layers.Conv2D(
            32, (3, 3), activation='swish', padding='same', name='block1_conv2')(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x)

        # Block 2
        x = layers.Conv2D(
            64, (3, 3), activation='swish', padding='same', name='block2_conv1')(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Conv2D(
            64, (3, 3), activation='swish', padding='same', name='block2_conv2')(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x)

        # Block 3
        x = layers.Conv2D(
            128, (3, 3), activation='swish', padding='same', name='block3_conv1')(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Conv2D(
            128, (3, 3), activation='swish', padding='same', name='block3_conv2')(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Conv2D(
            128, (3, 3), activation='swish', padding='same', name='block3_conv3')(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x)

        # Block 4
        x = layers.Conv2D(
            256, (3, 3), activation='swish', padding='same', name='block4_conv1')(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Conv2D(
            256, (3, 3), activation='swish', padding='same', name='block4_conv2')(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Conv2D(
            256, (3, 3), activation='swish', padding='same', name='block4_conv3')(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x)

        # Block 5
        x = layers.Conv2D(
            512, (3, 3), activation='swish', padding='same', name='block5_conv1')(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Conv2D(
            512, (3, 3), activation='swish', padding='same', name='block5_conv2')(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.Conv2D(
            512 , (3, 3), activation='swish', padding='same', name='block5_conv3')(x)
        x = layers.BatchNormalization(axis=chanDim)(x)
        x = layers.MaxPooling2D((2, 2), strides=(2, 2), name='block5_pool')(x)

        if include_top:
            # Classification block
            x = layers.Flatten(name='flatten')(x)
            x = layers.Dense(512, activation='swish', name='fc1')(x)
            x = layers.Dense(512, activation='swish', name='fc2')(x)
            x = layers.Dense(512, activation='swish', name='fc3')(x)
            x = layers.Dense(512, activation='swish', name='fc4')(x)


            x = layers.Dense(64, activation='softmax',
                             name='predictions')(x)



        model = Model(inputs=input, outputs=x, name='miniVGG' )

        return model





# Open file .txt save sequence of links to image train
print("[INFO] Loading images...")
f1 = open('/home/quan/PycharmProjects/sign-language-gesture-recognition/utils/filename_train.txt', 'r')
f2 = open('/home/quan/PycharmProjects/sign-language-gesture-recognition/utils/filename_2hand.txt', 'r')
lists_1 = f1.readlines()
lists_2 = f2.readlines()


links_images_1 = [i.split('\n')[0] for i in lists_1]
links_images_2 = [i.split('\n')[0] for i in lists_2]
links_images_2 = links_images_2[:36000]

for i in links_images_2:
    links_images_1.append(i)

del links_images_2

# extract the class labels from image paths then encode label
images = [i.split('/')[-1] for i in links_images_1]
labels = [int(p.split('_')[0]) for p in images]

del images




# load network
print("[INFO] loading network...")
model = miniVGG.build(96, 96, 1)

traingen=ImageDataGenerator()



# Start training model
model.compile(loss='sparse_categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

lrr = ReduceLROnPlateau(monitor='val_accuracy',patience=2,factor=0.2, min_lr=0.0001)


# extract batch of images and label then initialize list of actual image
batchPaths = links_images_1[:40000]
batchLabels = labels[:40000]
batchImages = []

#Preprocess image data
for (j, imagePath) in tqdm.tqdm(enumerate(batchPaths)):
    image = load_img(imagePath, target_size=(96, 96), color_mode='grayscale')
    image = img_to_array(image)
    image = np.expand_dims(image, axis=0)
    image = preprocess_input(image)
    batchImages.append(image)

# train model as feature extractor
batchImages = np.vstack(batchImages)
trainX, testX, trainY, testY = train_test_split(batchImages, batchLabels, random_state=42, test_size=0.2, shuffle=True)
del batchImages
del batchLabels
del batchPaths
del links_images_1


model.fit(traingen.flow(trainX,trainY, batch_size=32), validation_data=(testX, testY), callbacks=lrr, epochs=1)
# model.save("/home/quan/PycharmProjects/sign-language-gesture-recognition/models/sign_customized-vgg_3.model")







