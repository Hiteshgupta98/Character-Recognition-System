#!/usr/bin/env python
# coding: utf-8

# In[ ]:


pwd


# In[ ]:


import os 
import numpy as np
import cv2
import tensorflow as tf


# In[ ]:


CATEGORIES = os.listdir(r"C:\Users\hites\Desktop\ELC Internship\Training Data Images") 
#getting the labels, on which the model was trained


# In[ ]:


CATEGORIES


# ## Image Pre-processing

# In[ ]:


def image_alteration(image_path):
    IMG_SIZE = 128
    img_array = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    new_array = cv2.resize(img_array, (IMG_SIZE,IMG_SIZE))
    return new_array.reshape(-1,IMG_SIZE,IMG_SIZE,1)

model = tf.keras.models.load_model('ABCDE.model') #loading the saved model, which was trained earlier


# ## Output Function

# In[ ]:


def output_fxn(image_path):
    prediction = model.predict([image_alteration(image_path)])  #image given in the form of numpy arrays
    #prediction comes out to be an array, converting it to a list
    output = list(prediction[0])
    for i in range(0,len(output)):
        if(output[i] == 1 ): #checking what the model has predicted and mapping with CATEGORIES LIST
            index = i
            break
    print(CATEGORIES[index])


# In[ ]:


output_fxn('sample.png') 
#for predicting just give the path of the image 
#make sure it is in the present working directory or give absolute path for image
#it will predict the output :)


# In[ ]:




