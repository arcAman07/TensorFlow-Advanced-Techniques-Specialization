#!/usr/bin/env python
# coding: utf-8

# 
# 
# # Week 1: Multiple Output Models using the Keras Functional API
# 
# Welcome to the first programming assignment of the course! Your task will be to use the Keras functional API to train a model to predict two outputs. For this lab, you will use the **[Wine Quality Dataset](https://archive.ics.uci.edu/ml/datasets/Wine+Quality)** from the **UCI machine learning repository**. It has separate datasets for red wine and white wine.
# 
# Normally, the wines are classified into one of the quality ratings specified in the attributes. In this exercise, you will combine the two datasets to predict the wine quality and whether the wine is red or white solely from the attributes. 
# 
# You will model wine quality estimations as a regression problem and wine type detection as a binary classification problem.
# 
# #### Please complete sections that are marked **(TODO)**

# ## Imports

# In[1]:


import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
import itertools

import utils


# ## Load Dataset
# 
# 
# You will now load the dataset from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/index.php) which are already saved in your workspace.
# 
# ### Pre-process the white wine dataset (TODO)
# You will add a new column named `is_red` in your dataframe to indicate if the wine is white or red. 
# - In the white wine dataset, you will fill the column `is_red` with  zeros (0).

# In[2]:


## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.



# # URL of the white wine dataset
URI = './winequality-white.csv'

# # load the dataset from the URL
white_df = pd.read_csv(URI, sep=";")

# # fill the `is_red` column with zeros.
white_df["is_red"] = 0

# # keep only the first of duplicate items
white_df = white_df.drop_duplicates(keep='first')


# In[3]:


# You can click `File -> Open` in the menu above and open the `utils.py` file 
# in case you want to inspect the unit tests being used for each graded function.

utils.test_white_df(white_df)


# In[4]:


print(white_df.alcohol[0])
print(white_df.alcohol[100])

# EXPECTED OUTPUT
# 8.8
# 9.1


# ### Pre-process the red wine dataset (TODO)
# - In the red wine dataset, you will fill in the column `is_red` with ones (1).

# In[5]:


## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.



# # URL of the red wine dataset
URI = './winequality-red.csv'

# # load the dataset from the URL
red_df = pd.read_csv(URI, sep=";")

# # fill the `is_red` column with ones.
red_df["is_red"] = 1

# # keep only the first of duplicate items
red_df = red_df.drop_duplicates(keep='first')


# In[6]:


utils.test_red_df(red_df)


# In[7]:


print(red_df.alcohol[0])
print(red_df.alcohol[100])

# EXPECTED OUTPUT
# 9.4
# 10.2


# ### Concatenate the datasets
# 
# Next, concatenate the red and white wine dataframes.

# In[8]:


df = pd.concat([red_df, white_df], ignore_index=True)


# In[9]:


print(df.alcohol[0])
print(df.alcohol[100])

# EXPECTED OUTPUT
# 9.4
# 9.5


# In a real-world scenario, you should shuffle the data. For this assignment however, **you are not** going to do that because the grader needs to test with deterministic data. If you want the code to do it **after** you've gotten your grade for this notebook, we left the commented line below for reference

# In[10]:


#df = df.iloc[np.random.permutation(len(df))]


# This will chart the quality of the wines.

# In[11]:


df['quality'].hist(bins=20);


# ### Imbalanced data (TODO)
# You can see from the plot above that the wine quality dataset is imbalanced. 
# - Since there are very few observations with quality equal to 3, 4, 8 and 9, you can drop these observations from your dataset. 
# - You can do this by removing data belonging to all classes except those > 4 and < 8.

# In[12]:


## Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
## You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.



# # get data with wine quality greater than 4 and less than 8
df = df[(df['quality'] > 4) & (df['quality'] < 8 )]

# # reset index and drop the old one
df = df.reset_index(drop=True)


# In[13]:


utils.test_df_drop(df)


# In[14]:


print(df.alcohol[0])
print(df.alcohol[100])

# EXPECTED OUTPUT
# 9.4
# 10.9


# You can plot again to see the new range of data and quality

# In[15]:


df['quality'].hist(bins=20);


# ### Train Test Split (TODO)
# 
# Next, you can split the datasets into training, test and validation datasets.
# - The data frame should be split 80:20 into `train` and `test` sets.
# - The resulting `train` should then be split 80:20 into `train` and `val` sets.
# - The `train_test_split` parameter `test_size` takes a float value that ranges between 0. and 1, and represents the proportion of the dataset that is allocated to the test set.  The rest of the data is allocated to the training set.

# In[18]:


# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.



# Please do not change the random_state parameter. This is needed for grading.

# split df into 80:20 train and test sets
train, test = train_test_split(df, test_size=0.2, random_state = 1)
                               
# split train into 80:20 train and val sets
train, val = train_test_split(train, test_size=0.2, random_state = 1)


# In[19]:


utils.test_data_sizes(train.size, test.size, val.size)


# Here's where you can explore the training stats. You can pop the labels 'is_red' and 'quality' from the data as these will be used as the labels
# 

# In[20]:


train_stats = train.describe()
train_stats.pop('is_red')
train_stats.pop('quality')
train_stats = train_stats.transpose()


# Explore the training stats!

# In[21]:


train_stats


# ### Get the labels (TODO)
# 
# The features and labels are currently in the same dataframe.
# - You will want to store the label columns `is_red` and `quality` separately from the feature columns.  
# - The following function, `format_output`, gets these two columns from the dataframe (it's given to you).
# - `format_output` also formats the data into numpy arrays. 
# - Please use the `format_output` and apply it to the `train`, `val` and `test` sets to get dataframes for the labels.

# In[22]:


def format_output(data):
    is_red = data.pop('is_red')
    is_red = np.array(is_red)
    quality = data.pop('quality')
    quality = np.array(quality)
    return (quality, is_red)


# In[23]:


# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.



# format the output of the train set
train_Y = format_output(train)

# format the output of the val set
val_Y = format_output(val)
    
# format the output of the test set
test_Y = format_output(test)


# In[24]:


utils.test_format_output(df, train_Y, val_Y, test_Y)


# Notice that after you get the labels, the `train`, `val` and `test` dataframes no longer contain the label columns, and contain just the feature columns.
# - This is because you used `.pop` in the `format_output` function.

# In[25]:


train.head()


# ### Normalize the data (TODO)
# 
# Next, you can normalize the data, x, using the formula:
# $$x_{norm} = \frac{x - \mu}{\sigma}$$
# - The `norm` function is defined for you.
# - Please apply the `norm` function to normalize the dataframes that contains the feature columns of `train`, `val` and `test` sets.

# In[26]:


def norm(x):
    return (x - train_stats['mean']) / train_stats['std']


# In[28]:


# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.



# normalize the train set
norm_train_X = norm(train)
    
# normalize the val set
norm_val_X = norm(val)
    
# normalize the test set
norm_test_X = norm(test)


# In[29]:


utils.test_norm(norm_train_X, norm_val_X, norm_test_X, train, val, test)


# ## Define the Model (TODO)
# 
# Define the model using the functional API. The base model will be 2 `Dense` layers of 128 neurons each, and have the `'relu'` activation.
# - Check out the documentation for [tf.keras.layers.Dense](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Dense)

# In[34]:


# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.



def base_model(inputs):
    
    # connect a Dense layer with 128 neurons and a relu activation
    x = Dense(128,activation='relu')(inputs)
    
    # connect another Dense layer with 128 neurons and a relu activation
    x = Dense(128,activation='relu')(x)
    return x
  


# In[35]:


utils.test_base_model(base_model)


# # Define output layers of the model (TODO)
# 
# You will add output layers to the base model. 
# - The model will need two outputs.
# 
# One output layer will predict wine quality, which is a numeric value.
# - Define a `Dense` layer with 1 neuron.
# - Since this is a regression output, the activation can be left as its default value `None`.
# 
# The other output layer will predict the wine type, which is either red `1` or not red `0` (white).
# - Define a `Dense` layer with 1 neuron.
# - Since there are two possible categories, you can use a sigmoid activation for binary classification.
# 
# Define the `Model`
# - Define the `Model` object, and set the following parameters:
#   - `inputs`: pass in the inputs to the model as a list.
#   - `outputs`: pass in a list of the outputs that you just defined: wine quality, then wine type.
#   - **Note**: please list the wine quality before wine type in the outputs, as this will affect the calculated loss if you choose the other order.

# In[38]:


# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.



def final_model(inputs):
    
    # get the base model
    x = base_model(inputs)

    # connect the output Dense layer for regression
    wine_quality = Dense(units='1', name='wine_quality')(x)

    # connect the output Dense layer for classification. this will use a sigmoid activation.
    wine_type = Dense(units='1', activation='sigmoid', name='wine_type')(x)

    # define the model using the input and output layers
    model = Model(inputs=inputs, outputs=[wine_quality,wine_type])

    return model


# In[39]:


utils.test_final_model(final_model)


# ## Compiling the Model
# 
# Next, compile the model. When setting the loss parameter of `model.compile`, you're setting the loss for each of the two outputs (wine quality and wine type).
# 
# To set more than one loss, use a dictionary of key-value pairs.
# - You can look at the docs for the losses [here](https://www.tensorflow.org/api_docs/python/tf/keras/losses#functions).
#     - **Note**: For the desired spelling, please look at the "Functions" section of the documentation and not the "classes" section on that same page.
# - wine_type: Since you will be performing binary classification on wine type, you should use the binary crossentropy loss function for it.  Please pass this in as a string.  
#   - **Hint**, this should be all lowercase.  In the documentation, you'll see this under the "Functions" section, not the "Classes" section.
# - wine_quality: since this is a regression output, use the mean squared error.  Please pass it in as a string, all lowercase.
#   - **Hint**: You may notice that there are two aliases for mean squared error.  Please use the shorter name.
# 
# 
# You will also set the metric for each of the two outputs.  Again, to set metrics for two or more outputs, use a dictionary with key value pairs.
# - The metrics documentation is linked [here](https://www.tensorflow.org/api_docs/python/tf/keras/metrics).
# - For the wine type, please set it to accuracy as a string, all lowercase.
# - For wine quality, please use the root mean squared error.  Instead of a string, you'll set it to an instance of the class [RootMeanSquaredError](https://www.tensorflow.org/api_docs/python/tf/keras/metrics/RootMeanSquaredError), which belongs to the tf.keras.metrics module.
# 
# **Note**: If you see the error message 
# >Exception: wine quality loss function is incorrect.
# 
# - Please also check your other losses and metrics, as the error may be caused by the other three key-value pairs and not the wine quality loss.

# In[44]:


# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.



inputs = tf.keras.layers.Input(shape=(11,))
rms = tf.keras.optimizers.RMSprop(lr=0.0001)
model = final_model(inputs)

model.compile(optimizer=rms, 
              loss = {'wine_type' : 'binary_crossentropy',
                      'wine_quality' : 'mse'
                     },
              metrics = {'wine_type' : 'accuracy',
                         'wine_quality': tf.keras.metrics.RootMeanSquaredError()
                       }
             )


# In[45]:


utils.test_model_compile(model)


# ## Training the Model (TODO)
# 
# Fit the model to the training inputs and outputs. 
# - Check the documentation for [model.fit](https://www.tensorflow.org/api_docs/python/tf/keras/Model#fit).
# - Remember to use the normalized training set as inputs. 
# - For the validation data, please use the normalized validation set.
# 
# **Important: Please do not increase the number of epochs below. This is to avoid the grader from timing out. You can increase it once you have submitted your work.**

# In[46]:


# Please uncomment all lines in this cell and replace those marked with `# YOUR CODE HERE`.
# You can select all lines in this code cell with Ctrl+A (Windows/Linux) or Cmd+A (Mac), then press Ctrl+/ (Windows/Linux) or Cmd+/ (Mac) to uncomment.



history = model.fit(norm_train_X, train_Y,
                    epochs = 40, validation_data=(norm_val_X, val_Y))


# In[47]:


utils.test_history(history)


# In[48]:


# Gather the training metrics
loss, wine_quality_loss, wine_type_loss, wine_quality_rmse, wine_type_accuracy = model.evaluate(x=norm_val_X, y=val_Y)

print()
print(f'loss: {loss}')
print(f'wine_quality_loss: {wine_quality_loss}')
print(f'wine_type_loss: {wine_type_loss}')
print(f'wine_quality_rmse: {wine_quality_rmse}')
print(f'wine_type_accuracy: {wine_type_accuracy}')

# EXPECTED VALUES
# ~ 0.30 - 0.38
# ~ 0.30 - 0.38
# ~ 0.018 - 0.036
# ~ 0.50 - 0.62
# ~ 0.97 - 1.0

# Example:
#0.3657050132751465
#0.3463745415210724
#0.019330406561493874
#0.5885359048843384
#0.9974651336669922


# ## Analyze the Model Performance
# 
# Note that the model has two outputs. The output at index 0 is quality and index 1 is wine type
# 
# So, round the quality predictions to the nearest integer.

# In[52]:


predictions = model.predict(norm_test_X)
quality_pred = predictions[0]
type_pred = predictions[1]


# In[53]:


print(quality_pred[0])

# EXPECTED OUTPUT
# 5.4 - 6.0


# In[54]:


print(type_pred[0])
print(type_pred[944])

# EXPECTED OUTPUT
# A number close to zero
# A number close to or equal to 1


# ### Plot Utilities
# 
# We define a few utilities to visualize the model performance.

# In[55]:


def plot_metrics(metric_name, title, ylim=5):
    plt.title(title)
    plt.ylim(0,ylim)
    plt.plot(history.history[metric_name],color='blue',label=metric_name)
    plt.plot(history.history['val_' + metric_name],color='green',label='val_' + metric_name)


# In[56]:


def plot_confusion_matrix(y_true, y_pred, title='', labels=[0,1]):
    cm = confusion_matrix(y_true, y_pred)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(cm)
    plt.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + labels)
    ax.set_yticklabels([''] + labels)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    fmt = 'd'
    thresh = cm.max() / 2.
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
          plt.text(j, i, format(cm[i, j], fmt),
                  horizontalalignment="center",
                  color="black" if cm[i, j] > thresh else "white")
    plt.show()


# In[57]:


def plot_diff(y_true, y_pred, title = '' ):
    plt.scatter(y_true, y_pred)
    plt.title(title)
    plt.xlabel('True Values')
    plt.ylabel('Predictions')
    plt.axis('equal')
    plt.axis('square')
    plt.plot([-100, 100], [-100, 100])
    return plt


# ### Plots for Metrics

# In[58]:


plot_metrics('wine_quality_root_mean_squared_error', 'RMSE', ylim=2)


# In[59]:


plot_metrics('wine_type_loss', 'Wine Type Loss', ylim=0.2)


# ### Plots for Confusion Matrix
# 
# Plot the confusion matrices for wine type. You can see that the model performs well for prediction of wine type from the confusion matrix and the loss metrics.

# In[60]:


plot_confusion_matrix(test_Y[1], np.round(type_pred), title='Wine Type', labels = [0, 1])


# In[61]:


scatter_plot = plot_diff(test_Y[0], quality_pred, title='Type')

