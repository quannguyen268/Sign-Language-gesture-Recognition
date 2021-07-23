from tensorflow.keras.preprocessing.image import img_to_array, load_img
from tensorflow.keras.applications.inception_v3 import preprocess_input
from tensorflow.keras import models,layers
from tensorflow.keras.models import Sequential, Model
from sklearn.preprocessing import LabelEncoder
import csv
import numpy as np




# print("[INFO] Loading images...")
f1 = open('/home/quan/PycharmProjects/sign-language-gesture-recognition/utils/filename_train.txt', 'r')
# f1 = open('/home/quan/PycharmProjects/sign-language-gesture-recognition/utils/filename_2hand.txt', 'r')
lists_1 = f1.readlines()
links_images_1 = [i.split('\n')[0] for i in lists_1]



#
# # extract the class labels from image paths then encode label
images = [i.split('/')[-1] for i in links_images_1]
labels = [int(p.split('_')[0]) for p in images]



# load network
print("[INFO] loading network...")
model = models.load_model('/home/quan/PycharmProjects/sign-language-gesture-recognition/models/sign_customized-vgg_3.model')


# take the output of Dense layer with 512 nodes computing
x = model.layers[-2].output
extractor = models.Model(inputs=model.input, outputs=x, name='extractor')

extractor.summary()



# extractor = Sequential()
#
#
# for layer in model.layers[:-4]:
#     extractor.add(layer)
# extractor.add(layers.GlobalAveragePooling2D())
# print(extractor.summary())
#
path_to_csv = '/home/quan/PycharmProjects/sign-language-gesture-recognition/data/feature_train.csv'
with open(path_to_csv, 'w') as csvfile:
    writer = csv.writer(csvfile)
    for i in range(0, len(images)):
        # extract batch of images and label then initialize list of actual image
        imagepath = links_images_1[i]
        data = [labels[i]]

        #load image and preprocess image data
        image = load_img(imagepath, target_size=(96, 96), color_mode='grayscale')
        image = img_to_array(image)
        image = np.expand_dims(image, axis=0)
        image = preprocess_input(image)

        # pass image through network and use outputs as actual features

        feature = extractor.predict(image)
        value = np.asarray(feature).reshape((feature.shape[1], feature.shape[0]))
        value = value.flatten()
        data.extend(value)
        writer.writerow(data)





