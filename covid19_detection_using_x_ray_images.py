#This code was run in google colab

import os

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg

#cloning the dataset from the github repository
! git clone https://github.com/sohailzaveri/xray-covid-recognition.git

#setting the path to the main dir
main_dir = "/content/xray-covid-recognition/Data"

#setting the path to the train dir
train_dir = os.path.join(main_dir, 'train')

#setting the path to the test dir
test_dir = os.path.join(main_dir, 'test')

train_covid_dir = os.path.join(train_dir, 'COVID19')
train_normal_dir = os.path.join(train_dir, 'NORMAL')
test_covid_dir = os.path.join(test_dir, 'COVID19')
test_normal_dir = os.path.join(test_dir, 'NORMAL')

train_covid_names = os.listdir(train_covid_dir)
train_normal_names = os.listdir(train_normal_dir)
test_covid_names = os.listdir(test_covid_dir)
test_normal_names = os.listdir(test_normal_dir)

# Plotting a grid of 16 images (8 images of Covid and 8 images of Normal)

rows, cols = 4, 4

#setting the figure size
fig = plt.gcf()
fig.set_size_inches(12, 12)

#getting the filenames from the covid & normal dir of the train dataset
covid_pictures = [os.path.join(train_covid_dir, filename) for filename in train_covid_names[:8]]
normal_pictures = [os.path.join(train_normal_dir, filename) for filename in train_normal_names[:8]]

print(covid_pictures, normal_pictures)

#merging the covid and normal list
merged_list = covid_pictures + normal_pictures
for i, image_path in enumerate(merged_list):
  data = image_path.split('/', 6)[6]
  sp = plt.subplot(rows, cols, i+1)
  sp.axis('Off')
  image = mpimg.imread(image_path)
  sp.set_title(data, fontsize = 10)
  plt.imshow(image, cmap = 'gist_yarg')


# generating training, testing and validation batches
# setting horizontal_flip = True
train_data_gen = ImageDataGenerator(rescale = 1.0/255,
                                    validation_split = 0.2,
                                    zoom_range = 0.2,
                                    horizontal_flip = True)

validation_data_gen = ImageDataGenerator(rescale = 1.0/255)

test_data_gen = ImageDataGenerator(rescale = 1.0/255)

train_generator = train_data_gen.flow_from_directory(train_dir, 
                                                     target_size = (150, 150),
                                                     subset = 'training',
                                                     batch_size = 32,
                                                     class_mode = 'binary')

validation_generator = train_data_gen.flow_from_directory(train_dir, 
                                                     target_size = (150, 150),
                                                     subset = 'validation',
                                                     batch_size = 32,
                                                     class_mode = 'binary')

test_generator = test_data_gen.flow_from_directory(test_dir, 
                                                     target_size = (150, 150),
                                                     batch_size = 32,
                                                     class_mode = 'binary')


# Creating the model

model = Sequential()

# convolutional layer
model.add(Conv2D(filters = 32, kernel_size = (5, 5), padding = 'same', activation = 'relu', input_shape = (150, 150, 3)))
# pooling layer
model.add(MaxPooling2D(pool_size = (2, 2)))
# dropout layer
model.add(Dropout(0.5))
# convolutional layer
model.add(Conv2D(filters = 64, kernel_size = (5, 5), padding = 'same', activation = 'relu'))
# pooling layer
model.add(MaxPooling2D(pool_size = (2, 2)))
# dropout layer
model.add(Dropout(0.5))
# Flatten layer
model.add(Flatten())
# dense layer
model.add(Dense(256, activation = 'relu'))
# dropout layer
model.add(Dropout(0.5))
model.add(Dense(1, activation = 'sigmoid'))
model.summary()

model.compile(Adam(lr = 0.001), loss = 'binary_crossentropy', metrics = ['accuracy'])

#training the model
history = model.fit(train_generator,
                    epochs = 30,
                    validation_data = validation_generator)

#plot graph between training and validation loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.legend(['Training', 'Validation'])
plt.title('Training and validation losses')
plt.xlabel('epoch')

#plot graph between training and validation accuarcy
plt.plot(history.history['accuracy'])
plt.plot(history.history['val_accuracy'])
plt.legend(['Training', 'Validation'])
plt.title('Training and validation accuracy')
plt.xlabel('epoch')

test_loss, test_accuracy = model.evaluate(test_generator)

from google.colab import files
from keras.preprocessing import image
uploaded = files.upload()
for filename in uploaded.keys():
  image_path = '/content/' + filename
  img = image.load_img(image_path, target_size = (150, 150))
  images = image.img_to_array(img)
  images = np.expand_dims(images, axis = 0)
  prediction = model.predict(images)
  print(filename)

  if prediction == 0:
    print('covid detected')
  else:
    print('normal')

