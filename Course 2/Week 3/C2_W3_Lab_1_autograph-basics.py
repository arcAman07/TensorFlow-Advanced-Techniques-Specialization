#!/usr/bin/env python
# coding: utf-8

# # Autograph: Basic
# In this ungraded lab, you will go through  some of the basics of autograph so you can explore what the generated code looks like.

# ## Imports

# In[1]:


import tensorflow as tf


# ## Addition in autograph
# You can use the `@tf.function` decorator to automatically generate the graph-style code as shown below:

# In[2]:


@tf.function
def add(a, b):
    return a + b


a = tf.Variable([[1.,2.],[3.,4.]])
b = tf.Variable([[4.,0.],[1.,5.]])
print(tf.add(a, b))

# See what the generated code looks like
print(tf.autograph.to_code(add.python_function))


# ## if-statements in autograph
# Control flow statements which are very intuitive to write in eager mode can look very complex in graph mode. You can see that in the next examples: first a simple function, then a more complicated one that involves lots of ops and conditionals (fizzbuzz).

# In[3]:


# simple function that returns the square if the input is greater than zero
@tf.function
def f(x):
    if x>0:
        x = x * x
    return x

print(tf.autograph.to_code(f.python_function))


# ## Fizzbuzz in autograph
# 
# You may remember implementing [fizzbuzz](http://wiki.c2.com/?FizzBuzzTest) in preparation for a coding interview.  
# - Imagine how much fun it would be if you were asked to impement the graph mode version of that code!  
# 
# Fortunately, you can just use `@tf.function` and then call `tf.autograph.to_code`!

# In[4]:


@tf.function
def fizzbuzz(max_num):
    counter = 0
    for num in range(max_num):
        if num % 3 == 0 and num % 5 == 0:
            print('FizzBuzz')
        elif num % 3 == 0:
            print('Fizz')
        elif num % 5 == 0:
            print('Buzz')
        else:
            print(num)
        counter += 1
    return counter

print(tf.autograph.to_code(fizzbuzz.python_function))

