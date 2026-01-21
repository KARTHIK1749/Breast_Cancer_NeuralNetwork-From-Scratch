from model import PsychoDoctor
import numpy as np

X = np.random.rand(100,9) ## 100 samples , 9 features

model = PsychoDoctor(input_dim=9)

Y = np.random.randint(0,2, size =(10 , 1)) ## random binary labels for first 10 smaples

A = model.forward_prop(X[:10]) ## fp on first 10 samples

loss_before =  model.computed_loss(Y , A)

model.backward_prop(Y)

A_after = model.forward_prop(X[:10]) ## fp on first 10 samples
loss_after = model.computed_loss(Y , A_after)

print("Loss before backpropagation: ", loss_before)
print("Loss after backpropagation: ", loss_after)

