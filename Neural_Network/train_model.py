import numpy as np
import pandas as pd
from model import PsychoDoctor

## loading the cleaned cancer dataset
data = pd.read_csv('../data/cleaned_cancer_data.csv')
print("Data Loaded Successfully")
print(f"Data Shape: {data.shape}")

## splitting into X and Y
X = data.drop('output', axis =1).values 
Y = data['output'].values.reshape(-1 , 1)

print(f"Features shape : {X.shape}")

## normalize input features
X = (X - X.mean(axis=0))/X.std(axis=0)

print(f"Normalized Features shape : {X.shape}")

model = PsychoDoctor(input_dim=X.shape[1] , alpha =0.01)

## training model on epochs
epochs = 1000
print("Starting training the model at 20 epochs")

for epoch in range(epochs):
    A = model.forward_prop(X)
    loss = model.computed_loss(Y , A)
    model.backward_prop(Y)
    
    if epoch % 50 == 0:
        preds = (A>0.5).astype(int)
        accuracy = np.mean(preds == Y)
        print("Pred mean:", A.mean())
        print(f"Epoch {epoch} | Loss : {loss: .4f} | Accuracy : {accuracy*100 : .2f}%")
        # print("Pred min/max:", A.min(), A.max())


