import numpy as np
import pandas as pd
import math
import seaborn as sns
import matplotlib.pyplot as plt
%matplotlib inline
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix

def sigmoid(z):
    return 1.0/(1+np.exp(-z))

def loss(y, y_hat):
    loss = -np.mean(y*(np.log(y_hat)) - (1-y)*np.log(1-y_hat))
    return loss

def gradients(X, y, y_hat):
    
    # X --> Input.
    # y --> true/target value.
    # y_hat --> hypothesis/predictions.
    # w --> weights (parameter).
    # b --> bias (parameter).
    
    # m-> number of training examples.
    m = X.shape[0]
    
    # Gradient of loss w.r.t weights.
    dw = (1/m)*np.dot(X.T, (y_hat - y))
    
    # Gradient of loss w.r.t bias.
    db = (1/m)*np.sum((y_hat - y)) 
    
    return dw, db

def normalize(X):
    
    # X --> Input.
    
    # m-> number of training examples
    # n-> number of features 
    m, n = X.shape
    
    # Normalizing all the n features of X.
    for i in range(n):
        X = (X - X.mean(axis=0))/X.std(axis=0)
        
    return X

def train(X, y, bs, epochs, lr):
    
    # X --> Input.
    # y --> true/target value.
    # bs --> Batch Size.
    # epochs --> Number of iterations.
    # lr --> Learning rate.
        
    # m-> number of training examples
    # n-> number of features 
    m, n = X.shape
    
    # Initializing weights and bias to zeros.
    w = np.zeros((n,1))
    b = 0
    
    # Reshaping y.
    y = y.values.reshape(m,1)
    
    # Normalizing the inputs.
    x = normalize(X)
    
    # Empty list to store losses.
    losses = []
    
    # Training loop.
    for epoch in range(epochs):
        for i in range((m-1)//bs + 1):
            
            # Defining batches. SGD.
            start_i = i*bs
            end_i = start_i + bs
            xb = X[start_i:end_i]
            yb = y[start_i:end_i]
            
            # Calculating hypothesis/prediction.
            y_hat = sigmoid(np.dot(xb, w) + b)
            
            # Getting the gradients of loss w.r.t parameters.
            dw, db = gradients(xb, yb, y_hat)
            
            # Updating the parameters.
            w -= lr*dw
            b -= lr*db
        
        # Calculating loss and appending it in the list.
        l = loss(y, sigmoid(np.dot(X, w) + b))
        losses.append(l)
        
    # returning weights, bias and losses(List).
    return w, b
    
def predict(X,w,b):
    
    # X --> Input.
    m, n = X.shape
    # Normalizing the inputs.
    x = normalize(X)
    
    # Calculating predictions/y_hat.
    preds = sigmoid(np.dot(X, w) + b)
    
    # Empty List to store predictions.
    pred_class = []
    # if y_hat >= 0.5 --> round up to 1
    # if y_hat < 0.5 --> round up to 1
    pred_class = [1 if i > 0.5 else 0 for i in preds]
    
    return np.array(pred_class).reshape(m,1)

def performance_measure(Y_original,Y_predicted):
    cm=confusion_matrix(Y_original, Y_predicted)
    TP = cm[0][0]
    FP = cm[0][1]
    FN = cm[1][0]
    TN = cm[1][1]
#     TP, FP, TN, FN
    accuracy = (TP + TN)/(TP+TN+FP+FN)
#     True Positive Rate
    TPR = TP/(TP+FN) 
#     True Negative Rate 
    TNR=TN/(TN+FP)
#     Positive Predictive Value
    PPV=TP/(TP+FP)
#     False Discovery Rate
    FDR=FP/(FP+TP)
#     F1 Score
    F1 = 2*TP/(2*TP+FP+FN)
    
    return accuracy,TPR,TNR,PPV,FDR,F1



# Dataset Preprocessing for Telco Customer Data

def dataset_cleaning(dataset):
    dummy_data=pd.get_dummies(dataset[["gender","Partner","Dependents","PhoneService","MultipleLines",
                                       "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection","TechSupport",
                                       "StreamingTV","StreamingMovies","Contract","PaperlessBilling","PaymentMethod"]],
                              drop_first=True)
    dataset_after_concat=pd.concat([dataset,dummy_data],axis=1)
    dataset_clear=dataset_after_concat.drop(["gender","tenure","customerID","Partner","Dependents","PhoneService","MultipleLines",
                                             "InternetService","OnlineSecurity","OnlineBackup","DeviceProtection",
                                             "TechSupport","StreamingTV","StreamingMovies","Contract","PaperlessBilling",
                                             "PaymentMethod","MonthlyCharges","TotalCharges"],axis=1)
    return dataset_clear


dataset = pd.read_csv('telco.csv')
dataset=dataset_cleaning(dataset)
# dataset

# Independent variable
X=dataset.drop("Churn",axis=1)
# Dependent variable.....The values we need to predict

# Y=dataset["Churn"]
Y = pd.get_dummies(dataset["Churn"],drop_first=True)
# X
# Y

X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.3, random_state=1)
# X_train
# X_test
# y_train
# y_test

w, b = train(X_train, y_train, bs=100, epochs=1000, lr=0.01)
# w
# b

y_hat=predict(X_train,w,b)
# y_hat

accuracy,TPR,TNR,PPV,FDR,F1=performance_measure(y_train,y_hat)

print(f'*********************Performance Measurement for Training dataset************************ \n'
    f'Accuracy:\t\t\t {accuracy} \n'
     f'True Positive Rate:\t\t {TPR}\n'
     f'True Negative Rate:\t\t {TNR}\n'
     f'Positive Predictive Value:\t {PPV}\n'
     f'False Discovery Rate:\t\t {FDR}\n'
     f'F1 Score:\t\t\t {F1}\n'
     )

w, b = train(X_test, y_test, bs=100, epochs=1000, lr=0.01)
y_hat=predict(X_test,w,b)

accuracy,TPR,TNR,PPV,FDR,F1=performance_measure(y_test,y_hat)

print(f'*********************Performance Measurement for Test dataset************* \n'
    f'Accuracy:\t\t\t {accuracy} \n'
     f'True Positive Rate:\t\t {TPR}\n'
     f'True Negative Rate:\t\t {TNR}\n'
     f'Positive Predictive Value:\t {PPV}\n'
     f'False Discovery Rate:\t\t {FDR}\n'
     f'F1 Score:\t\t\t {F1}\n'
     )