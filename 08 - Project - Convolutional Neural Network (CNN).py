#!/usr/bin/env python
# coding: utf-8

# # Project: Create a Convolutional Neural Network
# - We will create a model on the [CIFAR-10 dataset](https://www.cs.toronto.edu/%7Ekriz/cifar.html)

# ### Step 1: Import libraries

# In[1]:


import tensorflow as tf
from tensorflow.keras import datasets, layers, models
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
import matplotlib.pyplot as plt
get_ipython().run_line_magic('matplotlib', 'inline')


# ### Step 2: Download the CIFAR10 dataset
# - Excute the cell below

# In[2]:


(train_images, train_labels), (test_images, test_labels) = datasets.cifar10.load_data()


# ### Step 3: Normalize the pixels
# - Divide the **train_images** and **test_images** with 255 to normalize them between 0 and 1.

# In[3]:


train_images = train_images / 255.0
test_images = test_images / 255.0


# In[ ]:





# ### Step 4: Get the class names of the labels
# - Make a class name conversion.
#     - HINT: make a list with the name **class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']**
# - How to use the name conversion.
#     - **class_names[int(train_labels[index])]**
# - How to show an image
#     - **plt.imshow(train_images[index])**

# In[4]:


class_names = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']


# In[5]:


index = 1
plt.imshow(train_images[index])
class_names[int(train_labels[index])]


# ### Step 5: Create a model
# - Create a **Sequential** model
#     - **Conv2D** with 32 and (3, 3),  **activation='relu', input_shape=(32, 32, 3)**
#     - **MaxPooling2D** with (2, 2)
#     - **Conv2D** with 64 and (3, 3),  **activation='relu'**
#     - **MaxPooling2D** with (2, 2)
#     - **Conv2D** with 64 and (3, 3),  **activation='relu'**
#     - **Flatten**
#     - **Dense** with 64 nodes with **input_dim=4, activaition='relu'**
#     - **Dense** with 10 (the output node)**
# - Complie the model with **optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy']**

# In[6]:


model = Sequential()
model.add(layers.Conv2D(32, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, input_dim=4, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


# In[ ]:





# ### Step 6: Train the model
# - Fit the model with **train_images, train_labels, epochs=10** and **validation_data=(test_images, test_labels)**

# In[7]:


model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))


# In[ ]:





# ### Stpe 7: Test the model
# - Make predictions
#     - assign the predictions of test_images to a variable
# - How to test visually
#     - Assign **index=0**
#     - Plot the image with **plt.imshow(test_images[index])**
#     - See the label from prediction by mapping it from **class_names**

# In[8]:


y_pred = model.predict(test_images)


# In[9]:


index = 168
plt.imshow(test_images[index])
class_names[y_pred[index].argmax()]


# In[10]:


model.evaluate(test_images, test_labels, verbose=0)


# ### Step 8 (Optional): Improve the model
# - Try to play around with the model to improve the score

# In[11]:


model = Sequential()
model.add(layers.Conv2D(64, (3, 3), activation='relu', input_shape=(32, 32, 3)))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.MaxPooling2D((2, 2)))
model.add(layers.Conv2D(64, (3, 3), activation='relu'))
model.add(layers.Flatten())
model.add(layers.Dense(64, input_dim=4, activation='relu'))
model.add(layers.Dense(10))

model.compile(optimizer='adam', loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True), metrics=['accuracy'])


# In[12]:


model.fit(train_images, train_labels, epochs=10, validation_data=(test_images, test_labels))


# In[13]:


y_pred = model.predict(test_images)


# In[14]:


index = 168
plt.imshow(test_images[index])
class_names[y_pred[index].argmax()]


# In[15]:


model.evaluate(test_images, test_labels, verbose=0)

