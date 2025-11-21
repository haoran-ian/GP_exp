
import numpy as np
from deap import gp


#%%
class create_pset:
    def __init__(self, x):
        self.x = x 
    
    def __call__(self):
        pset = gp.PrimitiveSet('MAIN', 1)
        pset.addEphemeralConstant('rand_num', lambda: np.random.random()*9+1) #1
        pset.addTerminal(self.x_(), name='x_') #2
        pset.addTerminal(self.first_dv(), name='first_dv') #3
        pset.addTerminal(self.trans_dv(), name='trans_dv') #4
        pset.addEphemeralConstant('rot_mat', lambda: self.rot_mat()) #5
        pset.addTerminal(self.index_vec(), name='index_vec') #6
        pset.addEphemeralConstant('rand_mat', lambda: self.rand_mat()) #7
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
        self.pset = pset
        return self.pset

    #%%
    # 2 # decision vector
    def x_(self):
        return self.x

    # 3 # first decision variable
    def first_dv(self):
        return self.x[:,0].reshape(self.x[:,0].shape[0],1)
        # return self.x[:,0]
    
    # 4 # translated decision vector
    def trans_dv(self):
        return np.vstack((self.x[:,1:].ravel(), np.zeros((len(self.x), 1)).ravel())).T
    
    # 5 # rotated matrix
    def rot_mat(self):
        mat_rand = np.random.rand(self.x.shape[1], self.x.shape[1])
        return np.dot(self.x, mat_rand)
    
    # 6 # index vector
    def index_vec(self):
        ind_ = np.array(range(1, self.x.shape[1]+1))
        return ind_.reshape(len(ind_),1)
        # return np.array(range(1, self.x.shape[1]+1))
    
    # 7 # random matrix between 1 and 1.1
    def rand_mat(self):
        mat_rand = np.random.rand(len(self.x), 1)
        return 1+mat_rand/10
# END CLASS

#%%
# 1 # real number between 1 and 10
def rand_num():
    return np.random.random()*9+1
    
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
    return x / y

# 21 # negative
def neg(x):
    return -1 * x

# 22 # reciprocal
def reciprocal(x):
    return 1 / x

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
    return np.log(abs(x))

# 31 # exponent
def exp(x):
    return np.exp(x)

# 32 # sum of vector
def sum_vec(x):
    return np.sum(x, axis=1).reshape(len(x), 1)

# 33 # mean of vector
def mean_vec(x):
    return np.mean(x, axis=1).reshape(len(x), 1)

# 34 # cumulative sum of vetor
def cumsum_vec(x):
    return np.cumsum(x, axis=1).reshape(len(x), 1)

# 35 # product of vector
def prod_vec(x):
    return np.prod(x, axis=1).reshape(len(x), 1)

# 36 # maximum of vector
def amax_vec(x):
    return np.amax(x, axis=1).reshape(len(x), 1)

# TODO: step function
def step(x):
    return np.square(np.floor(x+0.5))

# TODO: tanh
def tanh(x):
    return np.tanh(2*np.pi*x)