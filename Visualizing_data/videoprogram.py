import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.cross_validation import train_test_split
from sklearn.metrics import accuracy_score

# visulaize the important characteristics of the dataset
import matplotlib.pyplot as plt

# step 1: read the data
dataframe_all = pd.read_csv("pml-training.csv")
num_rows = dataframe_all.shape[0]

# step 2: remove useless data
# count the number of missing elements (NaN) in each column
counter_nan = dataframe_all.isnull().sum()
counter_without_nan = counter_nan[counter_nan==0]
# remove the columns with missing elements
dataframe_all = dataframe_all[counter_without_nan.keys()]#.keys returns only the column names
# remove the first 7 columns which contain no discriminative information
dataframe_all = dataframe_all.ix[:,7:]
# the list of columns (the last column is the class label)
columns = dataframe_all.columns
print columns

# step 3: get features (x) and scale the features
# get x and convert it to numpy array
x = dataframe_all.ix[:,:-1].values
standard_scaler = StandardScaler()
x_std = standard_scaler.fit_transform(x)

# step 4: get class labels y and then encode it into number
# get class label data
y = dataframe_all.ix[:,-1].values
# encode the class label
class_labels = np.unique(y) #to obtain all the classes of the categorical variable y
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# step 5: split the data into training set and test set
test_percentage = 0.1
x_train, x_test, y_train, y_test = train_test_split(x_std, y, test_size = test_percentage, random_state = 0)

# t-distributed Stochastic Neighbor Embedding (t-SNE) visualization
from sklearn.manifold import TSNE
tsne = TSNE(n_components=2, random_state=0)
x_test_2d = tsne.fit_transform(x_test)

# scatter plot the sample points among 5 classes
markers=('s', 'd', 'o', '^', 'v')
color_map = {0:'red', 1:'blue', 2:'lightgreen', 3:'purple', 4:'cyan'}
plt.figure()
for idx, cl in enumerate(np.unique(y_test)):
    plt.scatter(x=x_test_2d[y_test==cl,0], y=x_test_2d[y_test==cl,1], c=color_map[idx], marker=markers[idx], label=cl) 
plt.xlabel('X in t-SNE')
plt.ylabel('Y in t-SNE')
plt.legend(loc='upper left')
plt.title('t-SNE visualization of test data')
plt.show()



m = x_test_2d[y_test=='A',0];























































import pandas as pd
import numpy as np
from sklearn import *
import matplotlib.pyplot as pyplot

#Setp1 Data download/load
dataframe_all = pd.read_csv("pml-training.csv")
num_rows = dataframe_all.shape[0]
#dataframe_all.keys() gives column names
#Step2 -Clean th data
#count the number of missing elements(NaN) in each col
counter_nan = dataframe_all.isnull().sum()
counter_without_nan = counter_nan[counter_nan == 0]
dataframe_all = dataframe_all[counter_without_nan.keys()] # using only the columns without NaN va;ues
dataframe_all = dataframe_all.ix[:,7:]# removing columns from 1 to 7 & selecting there after
dataframe_all

#Step 3: create feature vectors
x = dataframe_all.ix[:,:-1].values# selecting all rows and columns from first till last but one.and only taking
#values without column names , it is like a compunded list where within a list each list is a row of df
standard_scalar = StandardScalar() #normalzing features so that all of them can be in proportion which
#helps algorithm to perform better
x_std = standard_scalar.fit_tranform(x)

#to achieve Dimensionality reduction when number of features are more we use
#Distributed Stochastic Neighbhor embedding (T_SNE) it reduces the dimensions
#feature vector from n dimensions to 2.T-SNE actually creates a similarity matrix
#which represents the similarity between the feature vectors

#initilaze t distribution Stochastic Neighbhor embedding visualization
tnse = TNSE(n_components=2,random_state=0)
x_test_2D = tnse.fit_tranform(x_std)

#scatter plot of the sample points among 5 classes
markers = ('s','d','o','^','v')
color_map = {0:'red',1:'blue',2:'lightgreen',3:'purple',4:'cyan'}
plt.figure()
for idx,cl in enumerate(np.unique(x_test_2D)):
    plt.scatter(x = x_test_2D[cl,0],y = x_test_2D[cl,1],c = color_map[idx],marker = markers[idx],label = cl)
    plt.show()
