import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, Activation
import pickle 

# GETTING THE FILES WHICH WE PREPARED IN THE DATA PREPRATION FILE...
x = pickle.load(open('Dataset.pickle', 'rb'))
y = pickle.load(open('Labels.pickle', 'rb'))
        
#just checking if the dataset is correct or not, by plotting the array...
img = x[0]
import matplotlib.pyplot as plt
plt.imshow(img,cmap='gray')
plt.show()
print(y[0])


#Normalizing Data - Converting pixel values which range from 0-255 to 0-1.. helps in faster processing of the image

x = x/255.0


#CNN

model = Sequential() #most common model used
#adding layers to the Neural Network
model.add(  Conv2D(32, 3,3 , input_shape = (128,128,1))  )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, 3,3 ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(32))

model.add(Dense(5)) #alert - in this Dense layer takes parameter as the number of different classes you have
#suppose you have 7 classes namely a,b,c,d,e,f,g,h then pass the parameter '7'
model.add(Activation('softmax'))

model.compile(loss = 'sparse_categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])

model.fit(x,y,validation_split = 0.1 , batch_size = 15, epochs =10)

model.save('ABCDE.model') #saving the trained model, so that you don't have to train it again and again.
