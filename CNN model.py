#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf
import os
from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense, Dropout, Activation


# In[3]:


import pickle
x = pickle.load(open('D.pickle', 'rb'))
y = pickle.load(open('L.pickle', 'rb'))
        


# In[ ]:


img = x[0]
import matplotlib.pyplot as plt
plt.imshow(img,cmap='gray')
plt.show()
print(y[0])


# ### Normalizing Data

# In[5]:


x = x/255.0


# ## CNN

# In[6]:


model = Sequential()
#adding layers to the Neural Network
model.add(  Conv2D(32, 3,3 , input_shape = (128,128,1))  )
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Conv2D(32, 3,3 ))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size = (2,2)))

model.add(Flatten())
model.add(Dense(32))

model.add(Dense(5))
model.add(Activation('softmax'))

model.compile(loss = 'sparse_categorical_crossentropy' , optimizer = 'adam' , metrics = ['accuracy'])


# In[8]:


model.fit(x,y,validation_split = 0.1 , batch_size = 15, epochs =10)


# In[9]:


model.save('ABCDE.model')


# In[10]:


pwd


# In[ ]:




