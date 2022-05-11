from tensorflow.keras import datasets, layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator 
from tensorflow.keras.preprocessing import image
from tensorflow.keras.optimizers import RMSprop
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
warnings.filterwarnings("ignore")
import cv2 

train = ImageDataGenerator(rescale = 1./255,
                   shear_range = 0.3,
                   horizontal_flip = True,
                   zoom_range = 0.3)
val = ImageDataGenerator(rescale = 1./255,
                   shear_range = 0.3,
                   horizontal_flip = True,
                   zoom_range = 0.3)
test = ImageDataGenerator(rescale=1/255)


train_dataset = train.flow_from_directory(r"D:\classes_ms\Big_Data\mini_proj\train", target_size = (200, 200), batch_size = 3,color_mode = 'rgb',
                                                   class_mode = 'categorical')
val_dataset = val.flow_from_directory(r"D:\classes_ms\Big_Data\mini_proj\val", target_size = (200, 200), batch_size = 3,color_mode = 'rgb',
                                                   class_mode = 'categorical')
test_dataset = test.flow_from_directory(r"D:\classes_ms\Big_Data\mini_proj\test", target_size = (200, 200), batch_size = 3, color_mode = 'rgb',
                                                   class_mode = 'categorical')
from keras.models import Sequential
from keras.layers import *

model = Sequential()
model.add(Conv2D(32,(3,3),activation = 'relu', input_shape = (200,200,3)))
model.add(MaxPooling2D())

model.add(Conv2D(32,(3,3),activation = 'relu', input_shape = (200,200,3)))
model.add(MaxPooling2D())

model.add(Conv2D(64,(3,3),activation = 'relu', input_shape = (200,200,3)))
model.add(MaxPooling2D())

model.add(Flatten())
model.add(Dense(1024,activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(3,activation = 'softmax'))

model.summary()

model.compile(loss = 'categorical_crossentropy',
              optimizer = 'rmsprop',
              metrics = ['accuracy'])

history = model.fit_generator(train_dataset, validation_data = val_dataset, epochs = 20)

plt.figure()
plt.plot(history.history['accuracy'], color='b', label="Training accuracy")
plt.plot(history.history['val_accuracy'], color='r',label="Validation accuracy")
plt.yticks(np.arange(0, 1, 0.1))
plt.xticks(np.arange(1, 20,1))
plt.legend(loc='best')
plt.show()
plt.close()

loss, accuracy = model.evaluate(test_dataset)
print('Accuracy:',accuracy)

test_dir = r"D:\classes_ms\Big_Data\mini_proj\test_all"

for i in os.listdir(test_dir):
    img = image.load_img(test_dir+'//'+i, target_size = (200,200))
    X = image.img_to_array(img)
    X = np.expand_dims(X,axis=0)
    images = np.vstack([X])
    images /= 255.
    val = model.predict(images)
    temp = list(val[0])
    c = ['banana', 'orange','strawberry']
    label = temp.index(max(temp))
    plt.title(c[label])
    plt.imshow(img)
    plt.show()
    