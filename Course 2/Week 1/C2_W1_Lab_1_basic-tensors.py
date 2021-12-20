#!/usr/bin/env python
# coding: utf-8

# # Basic Tensors

# In this ungraded lab, you will try some of the basic operations you can perform on tensors.

# ## Imports

# In[1]:


try:
    # %tensorflow_version only exists in Colab.
    get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
    pass

import tensorflow as tf
import numpy as np


# ## Exercise on basic Tensor operations
# 
# Lets create a single dimension numpy array on which you can perform some operation. You'll make an array of size 25, holding values from 0 to 24.

# In[2]:


# Create a 1D uint8 NumPy array comprising of first 25 natural numbers
x = np.arange(0, 25)
x


# Now that you have your 1-D array, next you'll change that array into a `tensor`. After running the code block below, take a moment to inspect the information of your tensor.

# In[3]:


# Convert NumPy array to Tensor using `tf.constant`
x = tf.constant(x)
x


# As the first operation to be performed, you'll square (element-wise) all the values in the tensor `x`

# In[4]:


# Square the input tensor x
x = tf.square(x)
x


# One feature of tensors is that they can be reshaped. When reshpaing, make sure you consider dimensions that will include all of the values of the tensor.

# In[5]:


# Reshape tensor x into a 5 x 5 matrix. 
x = tf.reshape(x, (5, 5))
x


# Notice that you'll get an error message if you choose a shape that cannot be exactly filled with the values of the given tensor.  
# * Run the cell below and look at the error message
# * Try to change the tuple that is passed to `shape` to avoid an error.

# In[15]:


# Try this and look at the error
# Try to change the input to `shape` to avoid an error
tmp = tf.constant([1,2,3,4])
tf.reshape(tmp, shape=(2,2))


# Like reshaping, you can also change the data type of the values within the tensor. Run the cell below to change the data type from `int` to `float`

# In[16]:


# Cast tensor x into float32. Notice the change in the dtype.
x = tf.cast(x, tf.float32)
x


# Next, you'll create a single value float tensor by the help of which you'll see `broadcasting` in action

# In[17]:


# Let's define a constant and see how broadcasting works in the following cell.
y = tf.constant(2, dtype=tf.float32)
y


# Multiply the tensors `x` and `y` together, and notice how multiplication was done and its result.

# In[9]:


# Multiply tensor `x` and `y`. `y` is multiplied to each element of x.
result = tf.multiply(x, y)
result


# Re-Initialize `y` to a tensor having more values.

# In[10]:


# Now let's define an array that matches the number of row elements in the `x` array.
y = tf.constant([1, 2, 3, 4, 5], dtype=tf.float32)
y


# In[11]:


# Let's see first the contents of `x` again.
x


# Add the tensors `x` and `y` together, and notice how addition was done and its result.

# In[12]:


# Add tensor `x` and `y`. `y` is added element wise to each row of `x`.
result = x + y
result


# ### The shape parameter for tf.constant
# 
# When using `tf.constant()`, you can pass in a 1D array (a vector) and set the `shape` parameter to turn this vector into a multi-dimensional array.

# In[13]:


tf.constant([1,2,3,4], shape=(2,2))


# ### The shape parameter for tf.Variable
# 
# Note, however, that for `tf.Variable()`, the shape of the tensor is derived from the shape given by the input array.  Setting `shape` to something other than `None` will not reshape a 1D array into a multi-dimensional array, and will give a `ValueError`.

# In[14]:


try:
    # This will produce a ValueError
    tf.Variable([1,2,3,4], shape=(2,2))
except ValueError as v:
    # See what the ValueError says
    print(v)

