# ---
# jupyter:
#   jupytext:
#     cell_metadata_json: true
#     notebook_metadata_filter: markdown
#     text_representation:
#       extension: .py
#       format_name: light
#       format_version: '1.5'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

import pandas as pd
import numpy as np

# # Topics in Pandas
# **Stats 507, Fall 2021** 
#   

# ## Contents
#
# + [Windows operations](#Windows-operations)
# + [Pandas Idiom: Splitting](#Pandas-Idiom:-Splitting)
# + [Time Series](#Time-Series)
# + [Pandas pipeline](#Pandas-pipeline)
# + [Missing Data in Pandas](#Missing-Data-in-Pandas)
# + [Hierarchical Indexing](#Hierarchical-Indexing)
# + [Introduction to pandas.DataFrame.fillna()](#Introduction-to-pandas.DataFrame.fillna())
# + [Missing Data Cleaning](#Missing-Data-Cleaning)
# + [pandas.DataFrame.Insert()](#pandas.DataFrame.Insert())
# + [Pandas sort_values() tutorial](#Pandas-sort_values()-tutorial)





# # Windows operations
# *Tiejin Chen*; **tiejin@umich.edu**
#

# - In the region of data science, sometimes we need to manipulate
#   one raw with two raws next to it for every raw.
# - This is one kind of windows operation.
# - We define windows operation as an operation that
#   performs an aggregation over a sliding partition of values (from pandas' userguide)
# - Using ```df.rolling``` function to use the normal windows operation

rng = np.random.default_rng(9 * 2021 * 20)
n=5
a = rng.binomial(n=1, p=0.5, size=n)
b = 1 - 0.5 * a + rng.normal(size=n)
c = 0.8 * a + rng.normal(size=n)
df = pd.DataFrame({'a': a, 'b': b, 'c': c})
print(df)
df['b'].rolling(window=2).sum()

# ## Rolling parameter
# In ```rolling``` method, we have some parameter to control the method, And we introduce two:
# - center: Type bool; if center is True, Then the result will move to the center in series.
# - window: decide the length of window or the customed window

df['b'].rolling(window=3).sum()

df['b'].rolling(window=3,center=True).sum()

df['b'].rolling(window=2).sum()

# example of customed window

window_custom = [True,False,True,False,True]
from pandas.api.indexers import BaseIndexer
class CustomIndexer(BaseIndexer):
    def get_window_bounds(self, num_values, min_periods, center, closed):
        start = np.empty(num_values, dtype=np.int64)
        end = np.empty(num_values, dtype=np.int64)
        for i in range(num_values):
            if self.use_expanding[i]:
                start[i] = 0
                end[i] = i + 1
            else:
                start[i] = i
                end[i] = i + self.window_size
        return start, end
indexer1 = CustomIndexer(window_size=1, use_expanding=window_custom)
indexer2 = CustomIndexer(window_size=2, use_expanding=window_custom)

df['b'].rolling(window=indexer1).sum()

df['b'].rolling(window=indexer2).sum()

# ## Windows operation with groupby
# - ```pandas.groupby``` type also have windows operation method,
#   hence we can combine groupby and windows operation.
# - we can also use ```apply``` after we use ```rolling```

df.groupby('a').rolling(window=2).sum()


def test_mean(x):
    return x.mean()
df['b'].rolling(window=2).apply(test_mean)













# # Pandas Idiom: Splitting
# Sean Kelly, seankell@umich.edu
#
# + [Splitting to analyze data](#Splitting-to-analyze-data)
# + [Splitting to create new Series](#Splitting-to-create-new-Series)
# + [Takeaways](#Takeaways)
#

# ## Pandas Idiom: Splitting
#
# - A useful way to utilize data is by accessing individual rows or groups of 
# rows and operating only on those rows or groups.  
# - A common way to access rows is indexing using the `loc` or `iloc` methods 
# of the dataframe. This is useful when you know what row indices you'd like to
# access.  
# - However, it is often required to subset a given data set based on some 
# criteria that we want each row of the subset to meet.  
# - We will look at selecting subsets of rows by splitting data based on row 
# values and performing analysis or calculations after splitting.
#
# ## Splitting to analyze data
#
# - Using data splitting makes it simple to create new dataframes representing 
# subsets of the initial dataframes
# - Find the average of one column of a group defined by another column

t_df = pd.DataFrame(
    {"col0":np.random.normal(size=10),
     "col1":np.random.normal(loc=10,scale=100,size=10),
     "col2":np.random.uniform(size=10)}
    )
t_below_average_col1 = t_df[t_df["col1"] < 10]
t_above_average_col1 = t_df[t_df["col1"] >= 10]
print([np.round(t_above_average_col1["col0"].mean(),4),
      np.round(t_below_average_col1["col0"].mean(),4)])

# ## Splitting to create new Series
#
# - We can use this splitting method to convert columns to booleans based on 
# a criterion we want that column to meet, such as converting a continuous 
# random variable to a bernoulli outcome with some probability p.

p = 0.4
t_df["col0_below_p"] = t_df["col2"] < p
t_df

# ## Takeaways
#
# - Splitting is a powerful but simple idiom that allows easy grouping of data
# for analysis and further calculations.  
# - There are many ways to access specific rows of your data, but it is
# important to use the right tool for the job.  
# - More information on splitting can be found [here][splitting].  
#
# [splitting]: https://pandas.pydata.org/pandas-docs/stable/user_guide/cookbook.html#splitting











# # Time Series
# Name: Yu Chi
# UM email: yuchi@umich.edu

# - The topic I picked is Time Series in pandas, specifically about time zone
#  representation.
# - Pandas has simple functionality for performing resampling operations during
#  frequency conversion (e.g., converting secondly data into 5-minutely data).
# - This can be quite helpful in financial applications.
#
# - First we construct the range and how frequent we want to stamp the time.
# - `rng = pd.date_range("10/17/2021 00:00", periods=5, freq="D")`
# - In this example, the starting time is 00:00 on 10/17/2021, the frequency
#  is one day, and the period is 5 days long.
# - Now we can consstruct the time representation.
# - `ts = pd.Series(np.random.randn(len(rng)), rng)`
# - If we try printing out ts, it should look like the following:

rng = pd.date_range("10/17/2021 00:00", periods=5, freq="D")
ts = pd.Series(np.random.randn(len(rng)), rng)
print(ts)

# - Then we set up the time zone. In this example, I'll set it to UTC
#  (Coordinated Universal Time).
# - `ts_utc = ts.tz_localize("UTC")`
# - If we try printing out ts_utc, it should look like the following:

ts_utc = ts.tz_localize("UTC")
print(ts_utc)

# - If we want to know what the time is in another time zone, it can easily
#  done as the following:
# - In this example, I want to convert the time to EDT (US Eastern time).
# - `ts_edt = ts_utc.tz_convert("US/Eastern")`
# - Let's try printing out ts_edt:

ts_edt = ts_utc.tz_convert("US/Eastern")
print(ts_edt)











# # Pandas pipeline
#
# ## Overview
#
# Name: Jiale Zha
#
# Email: jialezha@umich.edu
#  
# - [About Pipeline](#About-Pipeline)
#
# - [API](#API)
#
# - [Examples](#Examples)
#
# - [Takeaways](#Takeaways)

#

# ## About Pipeline
#
# A common situation in our data analyses is that we need the output of a function to be one of the input of another function. Pipiline is just the concept for that situation as it means we could regard those functions as pipes and connect them, let the data stream go through them to get the final result.

#

# ## API
#
# The pipeline function in pandas could be used for Series and DataFrame, the general API for it is,
#
# `pandas.Series.pipe(func, *args, **kwargs)`
#
# `pandas.DataFrame.pipe(func, *args, **kwargs)`
#
# where the input parameter `func` is the function to apply next, `args` are positional arguments of the function, and `kwargs` is a dictionary of keyword arguments.

#

# ## Examples
#
# A very common example for pipeline is the computation of composition function, say if we want to compute the result of the following function, 
#
# `f_3(f_2(f_1(df), arg1=a), arg2=b, arg3=c)`

# A more readable code for the above function will be 
#
# `(df.pipe(f_1)                 
#     .pipe(f_2, arg1=a)         
#     .pipe(f_3, arg2=b, arg3=c)`

#

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

#

# ### Takeaways
#
# - Use pipe method to do the multi-step data processing
#
# - Combine the pipe method with the other basic method in pandas













# # Missing Data in Pandas
#
# Shihao Wu, PhD student in statistics
#
# Reference: [https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html](https://pandas.pydata.org/pandas-docs/stable/user_guide/missing_data.html)
#
# There are 4 "slides" for this topic.
#
#
# ## Missing data
# Missing data arises in various circumstances in statistical analysis. Consider the following example:

# generate a data frame with float, string and bool values
df = pd.DataFrame(
    np.random.randn(5, 3),
    index=["a", "c", "e", "f", "h"],
    columns=["1", "2", "3"],
)
df['4'] = "bar"
df['5'] = df["1"] > 0

# reindex so that there will be missing values in the data frame
df2 = df.reindex(["a", "b", "c", "d", "e", "f", "g", "h"])

df2


# The missing values come from unspecified rows of data.

# ## Detecting missing data
#
# To make detecting missing values easier (and across different array dtypes), pandas provides the <code>isna()</code> and <code>notna()</code> functions, which are also methods on Series and DataFrame objects:

df2["1"]


pd.isna(df2["1"])


df2["4"].notna()


df2.isna()


# ## Inserting missing data
#
# You can insert missing values by simply assigning to containers. The actual missing value used will be chosen based on the dtype.
#
# For example, numeric containers will always use <code>NaN</code> regardless of the missing value type chosen:

s = pd.Series([1, 2, 3])
s.loc[0] = None
s


# Because <code>NaN</code> is a float, a column of integers with even one missing values is cast to floating-point dtype. pandas provides a nullable integer array, which can be used by explicitly requesting the dtype:

pd.Series([1, 2, np.nan, 4], dtype=pd.Int64Dtype())


# Likewise, datetime containers will always use <code>NaT</code>.
#
# For object containers, pandas will use the value given:

s = pd.Series(["a", "b", "c"])
s.loc[0] = None
s.loc[1] = np.nan
s


# ## Calculations with missing data
#
# Missing values propagate naturally through arithmetic operations between pandas objects.

a = df2[['1','2']]
b = df2[['2','3']]
a + b


# Python deals with missing value for data structure in a smart way. For example:
#
# * When summing data, NA (missing) values will be treated as zero.
# * If the data are all <code>NA</code>, the result will be 0.
# * Cumulative methods like <code>cumsum()</code> and <code>cumprod()</code> ignore <code>NA</code> values by default, but preserve them in the resulting arrays. To override this behaviour and include <code>NA</code> values, use <code>skipna=False</code>.

df2


df2["1"].sum()


df2.mean(1)


df2[['1','2','3']].cumsum()


df2[['1','2','3']].cumsum(skipna=False)


# Missing data is ubiquitous. Dealing with missing is unavoidable in data analysis. This concludes my topic here.












# # Hierarchical Indexing
#
# - Shushu Zhang
# - shushuz@umich.edu
# - Reference is [here](https://pandas.pydata.org/pandas-docs/stable/user_guide/advanced.html) 
#
# Hierarchical / Multi-level indexing is very exciting as it opens the door to some quite sophisticated data analysis and manipulation, especially for working with higher dimensional data. In essence, it enables you to store and manipulate data with an arbitrary number of dimensions in lower dimensional data structures like Series (1d) and DataFrame (2d).

# ### Creating a MultiIndex (hierarchical index) object
# - The MultiIndex object is the hierarchical analogue of the standard Index object which typically stores the axis labels in pandas objects. You can think of MultiIndex as an array of tuples where each tuple is unique. 
# - A MultiIndex can be created from a list of arrays (using MultiIndex.from_arrays()), an array of tuples (using MultiIndex.from_tuples()), a crossed set of iterables (using MultiIndex.from_product()), or a DataFrame (using MultiIndex.from_frame()). 
# - The Index constructor will attempt to return a MultiIndex when it is passed a list of tuples.

# Constructing from an array of tuples
arrays = [
    ["bar", "bar", "baz", "baz", "foo", "foo", "qux", "qux"],
    ["one", "two", "one", "two", "one", "two", "one", "two"],
]
tuples = list(zip(*arrays))
tuples
index = pd.MultiIndex.from_tuples(tuples, names=["first", "second"])
index

# ### Manipulating the dataframe with MultiIndex
#
# - Basic indexing on axis with MultiIndex is illustrated as below.
# - The MultiIndex keeps all the defined levels of an index, even if they are not actually used.

# Use the MultiIndex object to construct a dataframe 
df = pd.DataFrame(np.random.randn(3, 8), index=["A", "B", "C"], columns=index)
print(df)
df['bar']

#These two indexing are the same
print(df['bar','one'])
print(df['bar']['one'])

print(df.columns.levels)  # original MultiIndex
print(df[["foo","qux"]].columns.levels)  # sliced

# ### Advanced indexing with hierarchical index
# - MultiIndex keys take the form of tuples. 
# - We can use also analogous methods, such as .T, .loc. 
# - “Partial” slicing also works quite nicely.

df = df.T
print(df)
print(df.loc[("bar", "two")])
print(df.loc[("bar", "two"), "A"])
print(df.loc["bar"])
print(df.loc["baz":"foo"])


# ### Using slicers
#
# - You can slice a MultiIndex by providing multiple indexers.
#
# - You can provide any of the selectors as if you are indexing by label, see Selection by Label, including slices, lists of labels, labels, and boolean indexers.
#
# - You can use slice(None) to select all the contents of that level. You do not need to specify all the deeper levels, they will be implied as slice(None).
#
# - As usual, both sides of the slicers are included as this is label indexing.

# +
def mklbl(prefix, n):
    return ["%s%s" % (prefix, i) for i in range(n)]


miindex = pd.MultiIndex.from_product(
    [mklbl("A", 4), mklbl("B", 2), mklbl("C", 4), mklbl("D", 2)]
)


micolumns = pd.MultiIndex.from_tuples(
    [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")], names=["lvl0", "lvl1"]
)


dfmi = (
    pd.DataFrame(
        np.arange(len(miindex) * len(micolumns)).reshape(
            (len(miindex), len(micolumns))
        ),
        index=miindex,
        columns=micolumns,
    )
    .sort_index()
    .sort_index(axis=1)
)

print(dfmi)
print(dfmi.loc[(slice("A1", "A3"), slice(None), ["C1", "C3"]), :])
# + [What is fillna method?](#What-is-fillna-method?)


# + [What parameter does this method have?](#What-parameter-does-this-method-have?)

# -









# + [How to use & Examples](#How-to-use-&-Examples) [markdown]
#
# # Introduction to pandas.DataFrame.fillna()
#
# **Xinfeng Liu(xinfengl@umich.edu)**
#
# ## What is fillna method?
#
# * pandas.DataFrame.fillna(value=None, method=None, axis=None, inplace=False, limit=None, downcast=None)
# * Used to Fill NA/NaN values using the specified method.
#
# ## What parameter does this method have?
#
# * value(=None by default)
#    * value to use to fill null values
#    * could be a scalar or a dict/series/dataframe of values specifying which value to use for each index
# * method(=None by default)
#    * method to use for filling null values
#    * 'bfill'/'backfill': fill the null from the next valid obersvation
#    * 'ffill'/'pad': fill the null from the previous valid obersvation
# * axis(=None by default)
#    * fill null values along index(=0) or along columns(=1)
# * inplace(=False by default)
#    * if = True, fill in-place, and will not create a copy slice for a column in a DataFrame
# * limit(=None by default)
#    * a integer used to specify the maximum number of consecutive NaN values to fill. If there's a gap with more than this number of consecutive NaNs, it will be partially filled
# * downcast(=None by default)
#    * a dictionary of item->dtype of what to downcast if possible
#
# ## How to use & Examples

# +
import pandas as pd
import numpy as np

#create a dataframe with NaN
# this data frame represents the apple's hourly sales vlue 
# from 9am to 4pm in a store and it has some null values
df = pd.DataFrame([[25, np.nan, 23, 25, np.nan, 22, 20],
                   [22, 24, 25, np.nan, 21.5, np.nan, 20],
                   [27, 24.5, 20, 21, 19.5, 25, 22],
                   [19.5, np.nan, 22, np.nan, 27, 26, 21],
                   [21, 25.5, 26, np.nan, 22, 22, np.nan],
                   [30, np.nan, np.nan, 26, 29, 27.5, 35],
                   [27, 28, 30, 35, 37, np.nan, np.nan]],
                  columns=['monday', 
                           'tuesday', 
                           'wednesday', 
                           'thursday', 
                           'friday', 
                           'saturday',
                           'sunday'])
df
# -

# Now will can fill the null value using fillna method
df.fillna(method='ffill', axis=1)

# In this example, we used the previous valid value to fill the null value along the column. This actually make sense becuase each day's sale's value during the same period should be similar to each other. Therefore, fill the null with the same value as the day before will not affact the mean or variance the whole data














# # Missing Data Cleaning
#
# #### Chen Liu
#
# *ichenliu@umich.edu*

#
#
# - No data value is stored for some variables in an observation.
# - Here is a small example dataset we'll use in these slides.

# +
example = pd.DataFrame({
    'Col_1' : [1, 2, 3, 4, np.nan],
    'Col_2' : [9, 8, 7, 6, 5]
})
print(example)

### Insert missing data
example.loc[0, 'Col_1'] = None
print(example)
# -

# ## Calculation with missing data
#
# - When summing data, NA (missing) values will be 
#   treated as $0$ defaultly.
# - When producting data, NA (missing) values will be 
#   treated as $1$ defaultly.
# - To override this behaviour and include NA (missing) 
#   values, use `skipna=False`.
# - Calculate by series, NA (missing) values will yield 
#   NA (missing) values in result.

# +
### Default
print(example.sum())
print(example.prod())

### Include NA values
print(example.sum(skipna=False))
print(example.prod(skipna=False))

### Calculate by series
print(example['Col_1'] + example['Col_2'])
# -

# ## Logical operations with missing data
# - Comparison operation will always yield `False`.
# - `==` can not be used to detect NA (missing) values.
# - `np.nan` is not able to do logical operations, while `pd.NA` can. 

# +
### Comparison operation
print(example['Col_1'] < 3)
print(example['Col_1'] >= 3)

### These codes will yield all-False series
print(example['Col_1'] == np.nan)
print(example['Col_1'] == pd.NA)

### Logical operations of pd.NA
print(True | pd.NA, True & pd.NA, False | pd.NA, False & pd.NA)
# -

# ## Detect and delete missing data
# - `isna()` will find NA (missing) values.
# - `dropna()` will drop rows having NA (missing) values.
# - Use `axis = 1` in `dropna()` to drop columns having NA (missing) values.

# +
### Detect NA values
print(example.isna())

### Drop rows / columns having NA values 
print(example.dropna())
print(example.dropna(axis=1))
# -

# ## Fill missing data
# - `fillna()` can fill in NA (missing) values with 
#   non-NA data in a couple of ways.
# - Use `method='pad / ffill'` in `fillna()` to fill gaps forward.
# - Use `method='bfill / backfill'` in `fillna()` to fill gaps backward.
# - Also fill with a PandasObject like `DataFrame.mean()`.

# +
### Fill with a single value
print(example.fillna(0))

### Fill forward
print(example.fillna(method='pad'))

### Fill backward
print(example.fillna(method='bfill'))

### Fill with mean
print(example.fillna(example.mean()))
# -












# # pandas.DataFrame.Insert()
#
# Micah Scholes
#
# mscholes@umich.edu

# - The insert command is used to insert a new column to an existing dataframe.
# - While the merge command can also add columns to a dataframe, it is better for organizing data. The insert command works for data that is already organized where a column just needs to be added.

# ## Args
#
# The arguments for Insert() are:
# - loc: an integer representing the index location where the new column should be inserted
# - column: the name of the column. This should be a unique column name unless duplicates are desired.
# - value: a list, array, series, int, etc. of values to populate the column.
# - allow_duplicates: default is False. If set to true, it will allow you to insert a column with a duplicate name to an existing column.

# ## Example

# + [Topic Title](#Topic-Title)
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3, 4, 5, 6, 7, 8], 'b': [3, 4, 5, 6, 7, 8, 9,
                                                        10]})
df.insert(2,'c',["a", "b", "c", "d", "e", "f", "g", "h"])
df

# +
# An error is raised if a duplicate column name is attempted to be inserted without setting allow_duplicates to true

df.insert(0,'a', [5, 6, 7, 8, 9,10, 11, 12])

df.insert(0,'a', [5, 6, 7, 8, 9,10, 11, 12], True)
df

# +
# Additionally, the values have to be the same length as the other columns, otherwise we get an error.

df.insert(0,'d', [5, 6, 7, 8, 9,10, 11])

df.insert(0,'d', 1)
df
# However if only 1 value is entered, it will populate the entire column with that value.
# -













# # Pandas sort_values() tutorial
#
# Alan Hedrick, ajhedri@umich.edu
#
# ## General overview
# - Sometimes, you may need to sort your data by column </li>
# - This can be done through using the sort_values() function through pandas </li>
# - Below is a code cell creating a data frame of rows corresponding to an individuals
#   name, age, ID number, and location</li>
# - The data frame will show it's initial state, and the be sorted by name</li>
#

dataframe = pd.DataFrame({"Name": ["Alan", "Smore's", "Sparrow", "Tonks", "Marina"],                          "Age": [22, 2, 1, 5, 21],                          "ID Num": [69646200, 20000000, 86753090, 48456002, 16754598],                          "Location": ["Michigan", "Michigan", "Michigan", "Texas", "Michigan"]})

print("Original dataframe")
print(dataframe)
print("")

print("Dataframe sorted by Name")
#sort the dataframe in alphabetical order by name
dataframe.sort_values(by='Name', inplace=True)
print(dataframe)

# ## Function breakdown
#
# - In order to call the function, you only need to fill the "by" parameter </li>
# - This parameter is set to the name of the column you wish to sort by</li>
# - You may be wondering what the "inplace" parameter is doing</li>
#   - sort_values() by default returns the sorted dataframe; however, it does not update the dataframe unless "inplace" is specified to be True</li>
# - Below is an example showing this fact, notice that without "inplace" the sorted dataframe must be set equal to another

print(dataframe)
print("")
dataframe.sort_values(by="Age")
print(dataframe)
#notice how the age column has not been sorted at all
print("")

new_df = dataframe.sort_values(by="Age")
print(new_df)
#it's been sorted!
#let's check the original again

print("")
print(dataframe)
print("")
dataframe.sort_values(by="Age", inplace=True)
print(dataframe)
#both are valid ways to use the function!

# ## Sorting in descending order
# - By default, sort_values() will sort columns in ascending order, but this can be easily changed
# - To do this, set the parameter, "ascending," to False

print(dataframe)
print("")
dataframe.sort_values(by="Age", inplace = True, ascending = False)
print(dataframe)
#now it's sorted by age in descending order!

# ## Sort by multiple columns
# - To do this, merely specify more columns such as in the example below
# - This can be useful when generating plots and tables to view specific data

# +
print(dataframe)
print("")

dataframe.sort_values(by=["ID Num", "Location"], inplace = True)
print(dataframe)


# +
def mklbl(prefix, n):
    return ["%s%s" % (prefix, i) for i in range(n)]


miindex = pd.MultiIndex.from_product(
    [mklbl("A", 4), mklbl("B", 2), mklbl("C", 4), mklbl("D", 2)]
)


micolumns = pd.MultiIndex.from_tuples(
    [("a", "foo"), ("a", "bar"), ("b", "foo"), ("b", "bah")], names=["lvl0", "lvl1"]
)


dfmi = (
    pd.DataFrame(
        np.arange(len(miindex) * len(micolumns)).reshape(
            (len(miindex), len(micolumns))
        ),
        index=miindex,
        columns=micolumns,
    )
    .sort_index()
    .sort_index(axis=1)
)

print(dfmi)
print(dfmi.loc[(slice("A1", "A3"), slice(None), ["C1", "C3"]), :])
# + [What is fillna method?](#What-is-fillna-method?)



# + [What parameter does this method have?](#What-parameter-does-this-method-have?)

# -




