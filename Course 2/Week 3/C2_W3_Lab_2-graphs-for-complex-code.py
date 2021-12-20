#!/usr/bin/env python
# coding: utf-8

# <a href="https://colab.research.google.com/github/https-deeplearning-ai/tensorflow-3-public/blob/main/Course%202%20-%20Custom%20Training%20loops%2C%20Gradients%20and%20Distributed%20Training/Week%203%20-%20Autograph/C2_W3_Lab_2-graphs-for-complex-code.ipynb" target="_parent"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/></a>

# # Autograph: Graphs for complex code
# 
# In this ungraded lab, you'll go through some of the scenarios from the lesson `Creating graphs for complex code`.

# ## Imports

# In[1]:


try:
    # %tensorflow_version only exists in Colab.
    get_ipython().run_line_magic('tensorflow_version', '2.x')
except Exception:
    pass

import tensorflow as tf


# As you saw in the lectures, seemingly simple functions can sometimes be difficult to write in graph mode. Fortunately, Autograph generates this complex graph code for us.
# 
# - Here is a function that does some multiplication and additon.

# In[2]:


a = tf.Variable(1.0)
b = tf.Variable(2.0)

@tf.function
def f(x,y):
    a.assign(y * b)
    b.assign_add(x * a)
    return a + b

print(f(1.0, 2.0))

print(tf.autograph.to_code(f.python_function))


# - Here is a function that checks if the sign of a number is positive or not.

# In[3]:


@tf.function
def sign(x):
    if x > 0:
        return 'Positive'
    else:
        return 'Negative or zero'

print("Sign = {}".format(sign(tf.constant(2))))
print("Sign = {}".format(sign(tf.constant(-2))))

print(tf.autograph.to_code(sign.python_function))


# - Here is another function that includes a while loop.

# In[4]:


@tf.function
def f(x):
    while tf.reduce_sum(x) > 1:
        tf.print(x)
        x = tf.tanh(x)
    return x

print(tf.autograph.to_code(f.python_function))


# - Here is a function that uses a for loop and an if statement.

# In[5]:


@tf.function
def sum_even(items):
    s = 0
    for c in items:
        if c % 2 > 0:
            continue
        s += c
    return s

print(tf.autograph.to_code(sum_even.python_function))


# ## Print statements
# 
# Tracing also behaves differently in graph mode. First, here is a function (not decorated with `@tf.function` yet) that prints the value of the input parameter.  `f(2)` is called in a for loop 5 times, and then `f(3)` is called.

# In[6]:


def f(x):
    print("Traced with", x)

for i in range(5):
    f(2)
    
f(3)


# If you were to decorate this function with `@tf.function` and run it, notice that the print statement only appears once for `f(2)` even though it is called in a loop.

# In[7]:


@tf.function
def f(x):
    print("Traced with", x)

for i in range(5):
    f(2)
    
f(3)


# Now compare `print` to `tf.print`.
# - `tf.print` is graph aware and will run as expected in loops. 
# 
# Try running the same code where `tf.print()` is added in addition to the regular `print`.
# - Note how `tf.print` behaves compared to `print` in graph mode.

# In[8]:


@tf.function
def f(x):
    print("Traced with", x)
    # added tf.print
    tf.print("Executed with", x)

for i in range(5):
    f(2)
    
f(3)


# ## Avoid defining variables inside the function
# 
# This function (not decorated yet) defines a tensor `v` and adds the input `x` to it.  
# 
# Here, it runs fine.

# In[9]:


def f(x):
    v = tf.Variable(1.0)
    v.assign_add(x)
    return v

print(f(1))


# Now if you decorate the function with `@tf.function`.
# 
# The cell below will throw an error because `tf.Variable` is defined within the function. The graph mode function should only contain operations.

# In[12]:


# @tf.function
# def f(x):
#     v = tf.Variable(1.0)
#     v.assign_add(x)
#     return v

# print(f(1))


# To get around the error above, simply move `v = tf.Variable(1.0)` to the top of the cell before the `@tf.function` decorator.

# In[13]:


# define the variables outside of the decorated function
v = tf.Variable(1.0)

@tf.function
def f(x):
    return v.assign_add(x)

print(f(5))

