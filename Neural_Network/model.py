import numpy as np

## creating a simple neural network model class

## this class will hnadle the forward and backward propagation
## as well as the weight updates
## The basic structure of a neural network model
## input layer[ X[n_features]] -> HL-1[64_neurons] -> HL-2[32_neurons] -> HL-3[16_neurons]-> output layer[1_neuron]

class PsychoDoctor:
    def __init__(self , input_dim , alpha = 0.001):
        ''' Architecture of neural network model : input(9 features) -> 64 -> 32 -> 16-> output(yes/no)'''

        np.random.seed(42) ## for reproductibility
        self.alpha = alpha

        self.params ={
            "w1" : np.random.randn(input_dim , 64) * np.sqrt(2/input_dim),
            "b1" : np.zeros((1,64)),
            "w2" : np.random.randn(64 , 32) * np.sqrt(2/64),
            "b2" : np.zeros((1,32)),
            "w3" : np.random.randn(32 , 16) * np.sqrt(2/32),
            "b3" : np.zeros((1,16)),
            "w4" : np.random.randn(16 , 1) * np.sqrt(2/16),
            "b4" : np.zeros((1,1)) 
        }

        ## cache the forward pass
        self.cache ={}

    ## Activation functions: ReLU and Sigmoid
    def ReLU(self , Z):
        return np.maximum(0,Z)
        
    def sigmoid(self , Z):
        return 1/(1 + np.exp(-Z))
    
    ## Forward prop 
    def forward_prop(self , X):

        self.cache["X"] = X

        ## layer 1
        Z1 = X @ self.params["w1"]+ self.params["b1"]
        A1 = self.ReLU(Z1)

        ## Layer 2
        Z2 = A1 @ self.params['w2'] + self.params["b2"]
        A2 = self.ReLU(Z2)

        ## Layer 3
        Z3 = A2 @ self.params['w3'] + self.params["b3"]
        A3 = self.ReLU(Z3)

        ## Output Layer
        Z4 = A3 @ self.params["w4"] + self.params["b4"]
        A4 = self.sigmoid(Z4)

        ## updating all the cache
        self.cache.update(
            {
                "Z1":Z1,"A1":A1,"Z2":Z2,"A2":A2,"Z3":Z3,"A3":A3,"Z4":Z4,"A4":A4
            }
        )

        return A4
    


    ## Binary cross entropy loss (Loss fxn)

    # def computed_loss(self , Y_true , Y_pred):
    #     m = Y_true.shape[0]

    #     loss = -np.mean( Y_true * np.log(Y_pred + 1e-8)+ (1 - Y_true)* np.log(1-Y_true + 1e-8))

    #     return loss
    def computed_loss(self, Y, A): ## fixes class/output imbalance
        m = Y.shape[0]
        
        pos_weight = np.sum(Y == 0) / np.sum(Y == 1)
        loss = -np.mean(pos_weight * Y * np.log(A + 1e-8)+ (1 - Y) * np.log(1 - A + 1e-8))
        
        return loss

    
    ## relu backward 
    def ReLU_backward(self ,dA , Z):
        dZ = dA.copy()
        dZ[Z<=0] = 0
        return dZ
    
    ## backward prop 
    def backward_prop(self, Y):
        m = Y.shape[0]

        ## unpacking cacche
        A4 = self.cache["A4"]
        A3 = self.cache["A3"]
        A2 = self.cache["A2"]
        A1 = self.cache["A1"]
        Z4 = self.cache["Z4"]
        Z2 = self.cache["Z2"]
        Z1 = self.cache["Z1"]
        Z3 = self.cache["Z3"]
        X = self.cache["X"]

        ## output layer (sigmoid + bce loss)

        dZ4 = A4 - Y ## der of loss wrt z4
        dw4 = 1/m * (A3.T @ dZ4)
        db4 = 1/m * np.sum(dZ4 , axis = 0 , keepdims = True)

        ## layer 3 
        dA3 = dZ4 @ self.params["w4"].T
        dZ3 = self.ReLU_backward(dA3 ,Z3)
        dw3 = 1/m * (A2.T @ dZ3)
        db3 = 1/m * np.sum(dZ3 , axis  = 0, keepdims= True)

        ## layer 2
        dA2 = dZ3 @ self.params["w3"].T
        dZ2 = self.ReLU_backward(dA2 , Z2)
        dw2 = 1/m * (A1.T @ dZ2)
        db2 = 1/m * np.sum(dZ2 , axis = 0 , keepdims = True)

        ## layer 1 
        dA1 = dZ2 @ self.params["w2"].T
        dZ1 = self.ReLU_backward(dA1 , Z1)
        dw1 = 1/m * (X.T @ dZ1)
        db1 = 1/m * np.sum(dZ1 , axis =0 , keepdims= True)

        ## gradients descent updation 
        self.params["w4"] -= self.alpha * dw4
        self.params["b4"] -= self.alpha * db4
        self.params["w3"] -= self.alpha * dw3
        self.params["b3"] -= self.alpha * db3
        self.params["w2"] -= self.alpha * dw2
        self.params["b2"] -= self.alpha * db2
        self.params["w1"] -= self.alpha * dw1
        self.params["b1"] -= self.alpha * db1