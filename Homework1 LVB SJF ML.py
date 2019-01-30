##NAMES
#Laura Bishop
#Sarah Fite

##REFERENCES
#Code from Class with modifications

#Python documentation
#https://docs.python.org/3/tutorial/

#Sklearn clfs documentation
#https://scikit-learn.org/stable/supervised_learning.html

#https://chrisalbon.com/machine_learning/

#stackoverflow...in general

##LESSON
# 1. Write a function to take a list or dictionary of clfs and hypers ie use logistic regression,
# each with 3 DIFFERENT SETS of hyper parrameters for each



#Imports

import numpy as np
from sklearn.linear_model import LogisticRegressionCV
from sklearn.model_selection import KFold
from sklearn.datasets import load_iris
from sklearn.metrics import accuracy_score # other metrics?
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold  #EDIT: I had to import KFold
import dict_digger
from sklearn import metrics as mt


#iris data used
X, y = load_iris(return_X_y=True)
xdata = (X,y)


#picked 3  CLFs
clfsList = [RandomForestClassifier, LogisticRegression, LogisticRegressionCV]


#build hyper parameter list, 3 different sets per CLF
clfDict = {'RandomForestClassifier': {"min_samples_split": [2,3,4]},
           'RandomForestClassifier': {"n_estimators": [10, 100],
                                      "bootstrap": [True, False]},
           'RandomForestClassifier': {"min_samples_leaf" : [1],
                                     "criterion" :'gini',
                                     "n_estimators" : [10]},

           'LogisticRegression': {"tol": [0.001,0.01,0.1]},
           'LogisticRegression': {"class_weight":'balanced', "solver" : 'lbfgs'},
           'LogisticRegression': {"random_state" : [0]},

           'LogisticRegressionCV': {"class_weight":'balanced'},
           'LogisticRegressionCV': {"scoring" :'roc_auc', "fit_intercept" : [True, False]},
           'LogisticRegressionCV': {"Cs" : [100], "penalty" : [10]}}


#based on code provided in class. added accuracy score
def run(a_clf, data, clf_hyper={}):
  #M, L, n_folds = data # unpack data containter
  M, L = data
  n_folds=8


  kf = KFold(n_splits=n_folds) # Establish the cross validation
  ret = {} # classic explicaiton of results

  for ids, (train_index, test_index) in enumerate(kf.split(M, L)):
    clf = a_clf(**clf_hyper) # unpack paramters into clf is they exist

    clf.fit(M[train_index], L[train_index])

    pred = clf.predict(M[test_index])

    #accurate = accuracy_score(L[test_index], pred)

    ret[ids]= {'clf': clf,
               'train_index': train_index,
               'test_index': test_index,
               'accuracy': accuracy_score(L[test_index], pred)}

    #print("accuracy digger", dict_digger.dig(ret[ids], 'accuracy'))
    acc = mt.accuracy_score(L[test_index], pred)
    conf = mt.confusion_matrix(L[test_index], pred)
    print(clfs, " Accuracy ", acc)
    print(clfs, " Confusion")
    print (conf)
    #print('accuracy non digger:', acc )


  return ret

def myClfHypers(clfsList):

    for clf in clfsList:

    #I need to check if values in clfsList are in clfDict
        clfString = str(clf)
        print("clf: ", clfString)

        for k1, v1 in clfDict.items():  # go through first level of clfDict
            if k1 in clfString:		# if clfString1 matches first level
                for k2,v2 in v1.items(): # go through the inner dictionary of hyper parameters
                    print(k2)			 # for each hyper parameter in the inner list..
                    for vals in v2:		 # go through the values for each hyper parameter
                        print(vals)		 # and show them...

                        #pdb.set_trace()

myClfHypers(clfsList)





#Run function in loop

#Use run function with a list and a for loop


for clfs in clfsList:
    results = run(clfs, xdata, clf_hyper={})
    print(results)





import matplotlib.pyplot as plt
# for naming the plots from Dan on Slack in class
filename_prefix = 'clf_Histograms_'
n = 8 #number of folds
# initialize the plot_num counter for incrementing in the loop below
plot_num = 1

# Adjust matplotlib subplots for easy terminal window viewing
left  = 0.125  # the left side of the subplots of the figure
right = 0.9    # the right side of the subplots of the figure
bottom = 0.1   # the bottom of the subplots of the figure
top = 0.6      # the top of the subplots of the figure
wspace = 0.2   # the amount of width reserved for space between subplots,
               # expressed as a fraction of the average axis width
hspace = 0.2   # the amount of height reserved for space between subplots,
               # expressed as a fraction of the average axis height

#create the histograms
for k1, v1 in clfDict.items():
    # for each key in our clfsAccuracyDict, create a new histogram with a given key's values
    fig = plt.figure(figsize =(20,10)) # This dictates the size of our histograms
    ax  = fig.add_subplot(1, 1, 1) # As the ax subplot numbers increase here, the plot gets smaller
    plt.hist(v1, facecolor='green', alpha=0.75) # create the histogram with the values
    ax.set_title(k1, fontsize=30) # increase title fontsize for readability
    ax.set_xlabel('Classifer Accuracy (By K-Fold)', fontsize=25) # increase x-axis label fontsize for readability
    ax.set_ylabel('Frequency', fontsize=25) # increase y-axis label fontsize for readability
    ax.xaxis.set_ticks(np.arange(0, 1.1, 0.1)) # The accuracy can only be from 0 to 1 (e.g. 0 or 100%)
    ax.yaxis.set_ticks(np.arange(0, n+1, 1)) # n represents the number of k-folds
    ax.xaxis.set_tick_params(labelsize=20) # increase x-axis tick fontsize for readability
    ax.yaxis.set_tick_params(labelsize=20) # increase y-axis tick fontsize for readability
    #ax.grid(True) # you can turn this on for a grid, but I think it looks messy here.

    # pass in subplot adjustments from above.

    plt.subplots_adjust(left=left, right=right, bottom=bottom, top=top, wspace=wspace, hspace=hspace)
    plot_num_str = str(plot_num) #convert plot number to string
    filename = filename_prefix + plot_num_str # concatenate the filename prefix and the plot_num_str
    plt.savefig(filename, bbox_inches = 'tight') # save the plot to the user's working directory
    plot_num = plot_num+1 # increment the plot_num counter by 1

plt.show()
#Collapse
