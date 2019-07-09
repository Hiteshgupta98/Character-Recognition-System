#!/usr/bin/env python
# coding: utf-8

# In[ ]:


# IMPORTING ALL THE NECESSARY LIBRARIES
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt
import pickle
import random


# In[ ]:


Data_dir = 'D:\ELC Internship\Training Data Images' #giving the absolute path for the images
CATEGORIES = os.listdir("D:\ELC Internship\Training Data Images") #lists all the folder/file names 
#these are the labels of the images
print(CATEGORIES)


# In[ ]:


# JUST FOR VISUALIZING THAT IT IS WORKING OR NOT
for category in CATEGORIES:
    path = os.path.join(Data_dir ,category)
    # os.path.join() , joins two paths together i.e D:\ELC Internship\Training Data Images\A (iteration 1)
    for img in os.listdir(path): # now we are inside one of the folder(A,B,C,D,E) ,eg:A which contains 793 images 
        img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
        plt.imshow(img_array ,  cmap = 'gray')
        plt.show()
        break
    break
    


# In[ ]:


img_array


# In[ ]:


IMG_SIZE = 50 #RESIZING ALL THE IMAGES SO, THE DATA IS NORMALIZED

new_array = cv2.resize(img_array , (IMG_SIZE,IMG_SIZE))
plt.imshow(new_array, cmap = 'gray')
plt.show()


# In[ ]:


training_data= [] # EMPTY LIST
def create_training_data():
    for category in CATEGORIES: # traversing through the list of categories
        path = os.path.join(Data_dir ,category)
        class_num = CATEGORIES.index(category)
        for img in os.listdir(path):#going inside the category  A folder which contain images of 'A'
            img_array = cv2.imread(os.path.join(path,img),cv2.IMREAD_GRAYSCALE)
            new_array = cv2.resize(img_array,(IMG_SIZE,IMG_SIZE))
            training_data.append([new_array,class_num])
create_training_data()


# In[ ]:


len(training_data)
random.shuffle(training_data) # Shuffling the training data
training_data[1]


# In[ ]:


X = []
Y = []


# In[ ]:


for features, label in training_data:
    X.append(features)
    Y.append(label)
X = np.array(X).reshape(-1,img_size,img_size,1)


# In[ ]:



pickle_out = open('Dataset.pickle', "wb")
pickle.dump(X,pickle_out)
pickle_out.close()


pickle_out = open('Labels.pickle', 'wb')
pickle.dump(Y,pickle_out)
pickle_out.close()


# In[ ]:


pickle_in = open('Labels.pickle' , 'rb')
X=pickle.load(pickle_in)


# In[ ]:


X[1]


# In[5]:


pwd


# In[ ]:





# In[ ]:




