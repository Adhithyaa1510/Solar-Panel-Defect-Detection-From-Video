
import numpy as np
#import pandas as pd

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Convolution2D
from tensorflow.keras.layers import MaxPooling2D
from tensorflow.keras.layers import Flatten
from keras.preprocessing.image import ImageDataGenerator

train_datagen=ImageDataGenerator(rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
test_datagen=ImageDataGenerator(rescale=1./255)

x_train=train_datagen.flow_from_directory(r"D:\KEC\Student Project and Papers\Final Year students Project Paper - 2022 batch IT - Hari, Maheshwaran and Dinesh\Solar_Panel_Defect_Identification Running as on 24-03-2022\Solar Image Data Set\train", target_size=(128,128), batch_size=16, class_mode="categorical")

x_test=train_datagen.flow_from_directory(r"D:\KEC\Student Project and Papers\Final Year students Project Paper - 2022 batch IT - Hari, Maheshwaran and Dinesh\Solar_Panel_Defect_Identification Running as on 24-03-2022\Solar Image Data Set\test", target_size=(128,128), batch_size=16, class_mode="categorical")

print(x_train.class_indices)

model=Sequential()

model.add(Convolution2D(16,(3,3), input_shape=(128,128,3), activation="relu"))

model.add(MaxPooling2D(pool_size=(2,2)))

model.add(Flatten())

model.add(Dense(units=128,activation="relu"))

model.add(Dense(units=2,activation="softmax"))

model.summary()

model.compile(loss="categorical_crossentropy", optimizer="adam",metrics=["accuracy"])

# steps_per_epoch=no.of images in train data/batch_size
#306/16=19.1
# validation steps=no.of images in test data/batch _size
#199/16=12.4

model.fit(x_train,steps_per_epoch=20,epochs=20, validation_data=x_test, validation_steps=13)

model.save('Solar_Panel_Defect_Identification.h5')

from tensorflow.keras.models import load_model
#from tensorflow.keras.models import Sequential
from keras.preprocessing import image

import numpy as np

mymodel=load_model(r"D:\KEC\Student Project and Papers\Final Year students Project Paper - 2022 batch IT - Hari, Maheshwaran and Dinesh\Solar_Panel_Defect_Identification Running as on 24-03-2022\solar_panel_defect_Identification\Solar_Panel_Defect_Identification.h5")

img=image.load_img(r"D:\KEC\Student Project and Papers\Final Year students Project Paper - 2022 batch IT - Hari, Maheshwaran and Dinesh\Solar_Panel_Defect_Identification Running as on 24-03-2022\Solar Image Data Set\test\bad_condition\69.jpeg",target_size=(128,128))
img

xx1=image.img_to_array(img)

#xx1

#xx1.shape

xx2=np.expand_dims(xx1,axis=0)

#xx2.shape

#pred=mymodel.predict_class(xx2)
pred=np.argmax(mymodel)

print(pred)

y=mymodel.predict(xx2)
pred=np.argmax(y,axis=1)
print(pred)

print(x_train.class_indices )

index=['bad_condition','good_condition']
result=str(index[pred[0]])
print(result)