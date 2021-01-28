# How to slice a numpy array in a non-continuous way

Today, I was found in the situation where I have a numpy array with n rows and I want to slice it in a way the following way:

```python
array_sliced = array[[0:m, m+2:n], :]
```

The above syntax is not correct and brings up a SyntaxError. So, how can we go about this? This is not a tricky problem, but for an elegant solution, we need to think a bit more.

The easiest way is to create a list of indices that we want to use for the slice:

```python
list_indices = range(n)
```

but this will give me all the indices. So, instead, we can do a list comprehension with an if statement:

```python 
list_indices = [i for i in range(10) if i != m+1]
```

Then, we can avoid creating an extra variable and we can very easily slice the array in the following way:

```python
array_sliced = array[[i for i in range(10) if i != m+1], :]
```

Let's also provide an example:

```python
>>> n = 5
>>> x = np.random.rand(n, n)
>>> x.shape
(5, 5)
>>> m = 3
>>> y = x[[i for i in range(n) if i != m+1], :]
>>> y.shape
(4, 5)
```
Of course, further conditions could be applied and different constructions could be used for the index list.