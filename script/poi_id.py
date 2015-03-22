#!/usr/bin/python

import sys
import pickle
sys.path.append("../tools/")

from feature_format import featureFormat, targetFeatureSplit
from tester import test_classifier, dump_classifier_and_data

### Task 1: Select what features you'll use.
### features_list is a list of strings, each of which is a feature name.
### The first feature must be "poi".
features_list = ['poi','bonus', 'exercised_stock_options', 'expenses', 'from_messages','from_poi_to_this_person','from_this_person_to_poi','other','restricted_stock','salary','shared_receipt_with_poi','to_messages']

### Load the dictionary containing the dataset
data_dict = pickle.load(open("final_project_dataset.pkl", "r") )
import pandas as pd
raw_data=pd.DataFrame(data_dict.values())
raw_data.index=data_dict.keys()
data_fillna=raw_data.drop('email_address',1).astype('float64').fillna(raw_data.drop('email_address',1).astype('float64').mean())
d={}
key=list(data_fillna.index)
for j in range(len(key)):
    v={}
    key_v=list(data_fillna.columns.values)
    for i in range(len(key_v)):
        value_v=list(data_fillna.loc[key[j]])
        v[key_v[i]]= value_v[i]
    d[key[j]]=v
### Task 2: Remove outliers
### Task 3: Create new feature(s)
### Store to my_dataset for easy export below.

my_dataset = d
    
my_dataset.pop('TOTAL',0)  # spreadsheet phenomenon
my_dataset.pop('LOCKHART EUGENE E',0)  # all data 0 or N/A -- not helpful

new_data=pd.DataFrame(my_dataset.values())[features_list]
new_data.index=my_dataset.keys()
new_data['new_total_stock']=new_data['exercised_stock_options']+new_data['restricted_stock']
new_dataset={}
key=list(new_data.index)
for j in range(len(key)):
    v={}
    key_v=list(new_data.columns.values)
    for i in range(len(key_v)):
        value_v=list(new_data.loc[key[j]])
        v[key_v[i]]= value_v[i]
    new_dataset[key[j]]=v
### Extract features and labels from dataset for local testing
data = featureFormat(my_dataset, features_list, sort_keys = True)
labels, features = targetFeatureSplit(data)
features_list_new= ['poi','bonus', 'exercised_stock_options', 'expenses', 'from_messages','from_poi_to_this_person','from_this_person_to_poi','other','restricted_stock','salary','shared_receipt_with_poi','to_messages','new_total_stock']
data_new = featureFormat(new_dataset, features_list_new, sort_keys = True)
labels_new, features_new = targetFeatureSplit(data_new)
from sklearn import preprocessing
scaler = preprocessing.MinMaxScaler()
features = scaler.fit_transform(features)
features_new = scaler.fit_transform(features_new)
from numpy import mean
from sklearn import cross_validation
from sklearn.metrics import accuracy_score, precision_score, recall_score
import sys

def evaluate_clf(clf, features, labels, num_iters=100, test_size=0.3):
    #print clf
    accuracy = []
    precision = []
    recall = []
    first = True
    for trial in range(num_iters):
        features_train, features_test, labels_train, labels_test =\
            cross_validation.train_test_split(features, labels, test_size=test_size)
        clf.fit(features_train, labels_train)
        predictions = clf.predict(features_test)
        accuracy.append(accuracy_score(labels_test, predictions))
        precision.append(precision_score(labels_test, predictions))
        recall.append(recall_score(labels_test, predictions))
        '''
        if trial % 10 == 0:
            if first:
                sys.stdout.write('\nProcessing')
            sys.stdout.write('.')
            sys.stdout.flush()
            first = False
        '''
    #print "done.\n"
  #  print "precision: {}".format(mean(precision))
   # print "recall:    {}".format(mean(recall))
    return mean(precision), mean(recall)

### Task 4: Try a varity of classifiers
### Please name your classifier clf for easy export below.
### Note that if you want to do PCA or other multi-stage operations,
### you'll need to use Pipelines. For more info:
### http://scikit-learn.org/stable/modules/pipeline.html

# Provided to give you a starting point. Try a varity of classifiers.
'''
### Task 5: Tune your classifier to achieve better than .3 precision and recall 
### using our testing script.
### Because of the small size of the dataset, the script uses stratified
### shuffle split cross validation. For more info: 
### http://scikit-learn.org/stable/modules/generated/sklearn.cross_validation.StratifiedShuffleSplit.html

test_classifier(clf, my_dataset, features_list)

### Dump your classifier, dataset, and features_list so 
### anyone can run/check your results.

dump_classifier_and_data(clf, my_dataset, features_list)
'''