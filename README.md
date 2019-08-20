# sp
stock prediction via machine learning

sp is a small example to show the possible usage of OLS and MLP on stock prediction. The multilayer perceptron is completly written from stratch with multiple dense layers, activation and optimization functions. 

The first model tries to predict the next weekly log-return of the S&P 500 using sequential (non-batch-style) OLS and MLP predictors.

Output of the model is the Mean-Absolute-Error of log-returns (MAE):
```
Sequential log-return prediction
--------------------------------
Null model (last timepoint):
NULL:         0.02326
OLS with cholesky decomposition:
MAE OLS (o1): 0.01618
MAE OLS (o2): 0.01619
MAE OLS (o4): 0.01617
MAE OLS (o8): 0.01617

MLP with adam optimizer:
MAE MLP (o1): 0.01617
MAE MLP (o2): 0.01617
MAE MLP (o4): 0.01618
MAE MLP (o8): 0.01619
```

The second model tries to soley predict the next weeks movement (up/down) via a MLP classifier. The model has 1 hidden layer with 32 units, ReLU activation function, adam optimizer and softmax output.

Output of the model is the probability of guessing the movement right:
```
Sequential Movement (Up/Down) prediction
----------------------------------------
Null model (last movement):
Estimated probability of guessing right: 0.5017

MLP-Classifier (o1): 0.5624
MLP-Classifier (o2): 0.5626
MLP-Classifier (o4): 0.5631
MLP-Classifier (o8): 0.5624
```
