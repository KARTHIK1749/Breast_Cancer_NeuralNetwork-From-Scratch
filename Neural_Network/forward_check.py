## testing the forward propagation before implementing in into the main model 

import numpy as np
from model import PsychoDoctor

X = np.random.rand(100,9) ## 100 samples , 9 features

model = PsychoDoctor(input_dim=9)
A = model.forward_prop(X[:10]) ## fp on first 10 samples
print(A.shape)
print(A)