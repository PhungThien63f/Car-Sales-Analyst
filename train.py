import numpy as np 

import preprocessing_function as pf 
import config 

# Training step 

#Load data 
data = pf.load_data(config.PATH_TO_DATASET)
print(data)

 #divide data set 
X_train, X_test, y_train, y_test = pf.divide_train_test(data,config.TARGET)

print(X_train.shape)

