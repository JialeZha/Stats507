import pandas as pd


# # Question 0

# ## Pandas pipeline

# ### Overview
# 
# Name: Jiale Zha
# 
# Email: jialezha@umich.edu
#  
# - About Pipeline
#
# - API
#
# - Examples
#
# - Takeaways

# ### About  Pipeline

# A common situation in our data analyses is that we need the output of a function to be one of the input of another function. Pipiline is just the concept for that situation as it means we could regard those functions as pipes and connect them, let the data stream go through them to get the final result.

# ### API

# The pipeline function in pandas could be used for Series and DataFrame, the general API for it is,
#
# `pandas.Series.pipe(func, *args, **kwargs)`
#
# `pandas.DataFrame.pipe(func, *args, **kwargs)`

# where the input parameter `func` is the function to apply next, `args` are positional arguments of the function, and `kwargs` is a dictionary of keyword arguments.

# ### Examples

# A very common example for pipeline is the computation of composition function, say if we want to compute the result of the following function, 
#
# `f_3(f_2(f_1(df), arg1=a), arg2=b, arg3=c)`

# A more readable code for the above function will be 
#
# `(df.pipe(f_1)                 
#     .pipe(f_2, arg1=a)         
#     .pipe(f_3, arg2=b, arg3=c)`

# In practice, if we have the following data, and we want to normalize it, we could use the pipe function to process it step by step.

data = pd.DataFrame(
    {'math':[96, 95, 25, 34],
     'stats': [88, 46, 23, 100],
    'computer': [86, 93, 34, 34]})
data

# We normalize the data by subtracting its mean and dividing its standard deviation.

(data
 # Compute the mean
 .pipe(pd.DataFrame.mean)
 # Subtract the mean, which is the 'other' parameter in the subtraction function
 .pipe((pd.DataFrame.sub, 'other'), data) 
 # Divided by the standard deviation of the original data
 .pipe(pd.DataFrame.div, data.std()))

# ### Takeaways
#
# - Use pipe method to do the multi-step data processing
#
# - Combine the pipe method with the other basic method in pandas
