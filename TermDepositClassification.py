import numpy as np
#from sklearn.datasets import load_iris
from sknn.mlp import Classifier, Layer
from sklearn import preprocessing
from sklearn.metrics import accuracy_score 
from sklearn import cross_validation
from sklearn.cross_validation import train_test_split
from sklearn.cross_validation import cross_val_score
#from sklearn import datasets
import pandas as pd
import warnings
#import csv
import datetime
data = pd.read_csv('C:/Users/abeme/Desktop/Artificial Intelligence/Final Project/A.I. Final Project - Client Classification/bank/bank-full.csv', sep = ",")

var_names = data.columns.tolist()

categs = ['job','marital','education','default','housing', 'loan','contact', 'month','poutcome','y',]
quantit = [i for i in var_names if i not in categs]

data_job= pd.get_dummies(data['job'])
data_marital= pd.get_dummies(data['marital'])
data_education= pd.get_dummies(data['education'])
data_default= pd.get_dummies(data['default'])
data_housing= pd.get_dummies(data['housing'])
data_loan= pd.get_dummies(data['loan'])
data_contact=pd.get_dummies(data['contact'])
data_month= pd.get_dummies(data['month'])
data_poutcome=pd.get_dummies(data['poutcome'])
data_y=pd.get_dummies(data['y'])

df1 = data[quantit]
df1=df1.fillna(df1.mean())
df1_names = df1.keys().tolist()
min_max_scaler = preprocessing.MinMaxScaler()
x_scaled = min_max_scaler.fit_transform(df1)
df1 = pd.DataFrame(x_scaled)
df1.columns = df1_names
#df1.drop('data_y', axis=1, inplace=True)
final_df = pd.concat([df1,
                     data_job,
                     data_marital,
                     data_education,
                     data_default,
                     data_housing,
                     data_loan,
                     data_contact,
                     data_month,
                     data_poutcome,
                     data_y], axis = 1)
                     

"""This code uses the iris dataset to creates test and training sets, normalize and scale those sets, and then pass
those sets through the classifier in order to predicit which type of iris we are looking at.
The code then shows our accuracy of prediction.""" 

#Ignore warnings that fall within the Depreciation Warning category.
warnings.simplefilter("ignore", category=DeprecationWarning)

#X_train is where our initial data is stored for xtraining set.  This is raw data. 
#X_trainn is just like x_train but now normalized-- it later gets scaled.
#X_test is where our x test dataset goes.  This is raw data.
#X_testn is just like x_test but now normalized-- it later gets scaled
#y_train is our initial raw data for the y data.
#y_test is the raw y data for the test set.

#import some data to play with. This data comes from the sklearn datasets.    
#iris = pd.read_csv('C:/Users/abeme/Desktop/Artificial Intelligence/Final Project/A.I. Final Project - Client Classification/bank/bank-full.csv', sep = ",")

iris = final_df


#print iris

iris.data= iris.iloc[:,0:15]
#print X.shape
iris.target= iris.ix[:,-1]



#iris.data = iris.drop(16,1)
#iris.target = iris.drop(1,data_poutcome)

#print iris.data

#print (iris.target.head())
"""Sample the training set while holding 20% of the data for testing.  
The random_state parameter is the random number generator.  
If none, then the random_state instance is set to np.random.  
The reason for setting random_state to zero in this case is to make the outcome consistent across calls"""  

X_train, X_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2, random_state=0)
#Both X_trainn and X_testn are scaling the input vectors individually to a unit norm-- in this case a unit norm of l2. 
X_trainn = preprocessing.normalize(X_train, norm='l2')
X_testn = preprocessing.normalize(X_test, norm='l2')
#Scale both X_trainn and X_testn  so they have a unit mean of zero and unit variance. 
X_trainn = preprocessing.scale(X_trainn)
X_testn = preprocessing.scale(X_testn)

""" The data in clsfr contains integer labels as outputs and we want to classify the data. The "Classifier" parameter helps
us do that.  Essentially we are setting up the model and then using various layer types and parameters to predict iris type""" 
clsfr = Classifier(   
	layers=[  #"""The layer parameters determines how the neural network is structured. #In this case the first two layers are structured as recififiers. 
    	Layer("Rectifier", units=13),   
    	Layer("Rectifier", units=13),
          #Softmax is the output layer activation type used in the output layer.  Softmax is the recommened default for classification problems.
    	Layer("Softmax")],    	
       # The learning_rate determines the speed at which the ANN arrives at the minimum solution. 
       learning_rate=0.01,
       # The learning rule determines how the ANN learns, in this case by Stochastic gradient descent, which is trying to find the minima or mazima by iteration.  
       learning_rule='sgd',
       #The random_state here is the random number generator.  Having it set to a number allows for the outcome to be consistent across calls. 
       random_state=201,
       #n_iter tells the code to train for 200 iterations.
	    n_iter=100)    
start_time = datetime.datetime.now()
#This line fits the classification model using X_trainn and y_train to model1
model1=clsfr.fit(X_trainn, y_train)
stop_time = datetime.datetime.now()
print "Time required for optimization:",stop_time - start_time
# This line sets the prediction of the the X_testn to y_hat. 
y_hat=clsfr.predict(X_testn)
"""This line estimates the accuracy of the clsfr on the X_train and y_train by splitting the X_train and computing the score 
5 consectutive times with different splits each time."""
scores = cross_val_score(clsfr, X_trainn, y_train, cv=5)
#Print scores
print scores
#Print the accuracy of the training mean of "scores".
print 'train mean accuracy %s' % np.mean(scores)
#Pritng the percentage of the accuracy_score of y_hat and y_test.
print 'vanilla sgd test %s' % accuracy_score(y_hat,y_test)