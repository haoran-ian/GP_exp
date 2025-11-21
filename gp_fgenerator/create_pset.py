
import numpy as np
from deap import gp


def create_pset():
    pset = gp.PrimitiveSet('MAIN', 1)
    pset.addEphemeralConstant('real_num', lambda: np.random.random()*9+1) #1
    # pset.addTerminal(x_(), name='x_') #2
    pset.addPrimitive(first_dv, 1) #3
    pset.addPrimitive(trans_dv, 1) #4
    # pset.addEphemeralConstant('rot_mat', lambda: np.random.rand(1,)) #5
    # pset.addTerminal(index_vec(x), name='index_vec') #6
    pset.addEphemeralConstant('rand_num', lambda: 1+np.random.random()/10) #7
    pset.addPrimitive(add, 2) #11
    pset.addPrimitive(sub, 2) #12
    pset.addPrimitive(mul, 2) #13
    pset.addPrimitive(div, 2) #14
    pset.addPrimitive(neg, 1) #21
    pset.addPrimitive(reciprocal, 1) #22
    pset.addPrimitive(mul10, 1) #23
    pset.addPrimitive(square, 1) #24
    pset.addPrimitive(sqrt, 1) #25
    pset.addPrimitive(abs_, 1) #26
    pset.addPrimitive(roundoff, 1) #27
    pset.addPrimitive(sin, 1) #28
    pset.addPrimitive(cos, 1) #29
    pset.addPrimitive(ln, 1) #30
    pset.addPrimitive(exp, 1) #31
    pset.addPrimitive(sum_vec, 1) #32
    pset.addPrimitive(mean_vec, 1) #33
    pset.addPrimitive(cumsum_vec, 1) #34
    pset.addPrimitive(prod_vec, 1) #35
    pset.addPrimitive(amax_vec, 1) #36
    pset.renameArguments(ARG0='x')
    return pset
# END DEF

#%%
# 1 # real number between 1 and 10
# def rand_num():
#     return np.random.random()*9+1

# 2 # decision vector
# def x_(self):
#     return self.x

# 3 # first decision variable
def first_dv(x):
    if (np.isscalar(x)):
        return x
    return x[0]

# 4 # translated decision vector
def trans_dv(x):
    if (np.isscalar(x)):
        return x
    return np.hstack((x[1:].ravel(), np.zeros((1, 1)).ravel()))

# 5 # rotated matrix
# def rot_mat(x):
#     mat_rand = np.random.rand(x.shape[1], x.shape[1])
#     return np.dot(x, mat_rand)

# 6 # index vector
# def index_vec(x):
#     return np.array(range(1, len(x)+1))

# 7 # random number between 1 and 1.1
# def rand_mat(x):
#     mat_rand = np.random.rand(len(x), 1)
#     return 1+mat_rand/10
    
# 11 # addition
def add(x, y):
    return x + y

# 12 # subtraction
def sub(x, y):
    return x - y

# 13 # multiplication
def mul(x, y):
    return x * y

# 14 # division
def div(x, y):
    return np.where(np.abs(y)>1e-20, np.divide(x, y), 1.)

# 21 # negative
def neg(x):
    return -1 * x

# 22 # reciprocal
def reciprocal(x):
    return np.where(np.abs(x)>1e-20, np.divide(1, x), 1.)

# 23 # multiply by 10
def mul10(x):
    return 10 * x

# 24 # square
def square(x):
    return np.square(x)

# 25 # square root
def sqrt(x):
    return np.sqrt(abs(x))

# 26 # absolute value
def abs_(x):
    return abs(x)

# 27 # rounded value
def roundoff(x):
    return np.round(x)

# 28 # sine
def sin(x):
    return np.sin(2*np.pi*x)

# 29 # cosine
def cos(x):
    return np.cos(2*np.pi*x)

# 30 # natural logarithm
def ln(x):
    # return np.log(x)
    return np.where(np.abs(x)>1e-20, np.log(abs(x)), 1.)

# 31 # exponent
def exp(x):
    return np.exp(x)

# 32 # sum of vector
def sum_vec(x):
    return np.sum(x)

# 33 # mean of vector
def mean_vec(x):
    return np.mean(x)

# 34 # cumulative sum of vetor
def cumsum_vec(x):
    return np.cumsum(x)

# 35 # product of vector
def prod_vec(x):
    return np.prod(x)

# 36 # maximum of vector
def amax_vec(x):
    return np.amax(x)

# TODO: power function
def power(x):
    return np.power(x)

# TODO: step function
def step(x):
    return np.square(np.floor(x+0.5))

# TODO: tangent
# def tan(x):
#     return np.tan(2*np.pi*x)