

import numpy as np
import matplotlib.pyplot as plt
import pandas
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import train_test_split # Import train_test_split function
from sklearn import metrics #Import scikit-learn metrics module for accuracy calculation
import pylab
import random



df = pandas.read_csv("FILEPATH.csv")
df.info()
#print(df['Purchase']) :: examples of how to view data in .csv
#print(df['Purchase'][2]):: examples of how to view data in .csv
d = {"No": 0, "Yes": 1}# No ==0 , Yes ==1
df['Purchase'] = df['Purchase'].map(d)# sets each to its proper value
features = ['Age','AnnualIncome','SpendingScore','DaysSinceLastPurchase','TotalPurchases']
X = df[features]
y = df['Purchase']

#TraininSet = range(1,634)# generalize code using a subset range for testing

Rstate = random.seed(3)# sets the random seed for reproducible results
#TraininSet = random.sample(range(1,1000), 650) # random sample w/o replacement
TraininSet = random.choices(range(1,1000), k=650) # random sample with replacement, bootstrapping


#TD = pandas.read_csv("TestData.txt", header = None) # can be used for another dataset as test samples, simulated samples etc.
#TestSet = random.sample(list(set(range(1, 1000)) - set(TraininSet)),(1000-651))# excludes unchosen smaples from training set// take total minus test sample minus one w/o replacement
TestSet = random.choices(list(set(range(1, 1000)) - set(TraininSet)),k=(1000-651))# excludes unchosen samaples from training set// take total minus test sample minus one with replacement
TD = X.loc[TestSet]# creates test dataset
TD.info()# displays contents of TD

# initialize accuacy list to search for depth with highest accuracy
AccuSearch = []

for i in range(1,20):# runs through different leels of pruning and measures accuracy between training and test sets

    Rstate = random.seed(3)

    ##Train Model with first subset of samples
    dtree = DecisionTreeClassifier(max_depth=i, random_state=Rstate)
    dtree = dtree.fit(X.loc[TraininSet], y.loc[TraininSet])

    # test with i[] using fit from first TrainingSet
    TX = dtree.predict(TD)
    TY = y.loc[TestSet]


    print("Accuracy for Depth[",i,"]"," = ",metrics.accuracy_score(TX,TY))
    AccuSearch.append(metrics.accuracy_score(TX,TY))
print("#############################")
print("#############################")
print(" Highest accuracy  = ", max(AccuSearch), " at Depth =[", AccuSearch.index(max(AccuSearch))+1,"]")# add 1 due to list properties
print("#############################")
print("#############################")

loadDepth = AccuSearch.index(max(AccuSearch))

print("Training X Set::: \n",X.loc[TraininSet])
print("Training Y Set::: \n",y.loc[TraininSet])
print("Testing Set::: \n",TD)


### TREE PLOTS
#dtree = DecisionTreeClassifier(max_depth=6, random_state=2)
#dtree = dtree.fit(X.loc[TraininSet], y.loc[TraininSet])
#fig1 = plt.figure(figsize=(15.0,15.0))
#tree.plot_tree(dtree, feature_names=features)
#plt.show()



## interating for different random states to generate multiple trees at constant pruning depth
## Choosing the index in AccuLIST with the max accuracy would ideally produce the best model. The index in this list is the random seed state
# which in turns pulls the samples used to create the most accurate model

print(" You might have to press ENTER a few times")

AccuLIST = []

N=10
Depth0 = loadDepth

for i in range(1,N):# runs through different leels of pruning and measures accuracy between training and test sets

    Rstate = random.seed(i)# sets the random seed for reproducible results
    TraininSet = random.sample(range(1,1000), 650) # random sample w/o replacement
    #TraininSet = random.choices(range(1,1000), k=650) # random sample with replacement, bootstrapping


    #TD = pandas.read_csv("TestData.txt", header = None) # can be used for another dataset as test samples, simulated samples etc.
    
    TestSet = random.sample(list(set(range(1, 1000)) - set(TraininSet)),(1000-651))# excludes unchosen smaples from training set// take total minus test sample minus one w/o replacement
    #TestSet = random.choices(list(set(range(1, 1000)) - set(TraininSet)),k=(1000-651))# excludes unchosen samaples from training set// take total minus test sample minus one with replacement
    TD = X.loc[TestSet]# creates test dataset
    #TD.info()# displays contents of TD

    ##Train Model with first subset of samples
    dtree = DecisionTreeClassifier(max_depth=Depth0, random_state=Rstate)
    dtree = dtree.fit(X.loc[TraininSet], y.loc[TraininSet])

    # test with i[] using fit from first TrainingSet
    TX = dtree.predict(TD)
    TY = y.loc[TestSet]

    # uncomment for small values of 'N'
    print("Accuracy at Random State[",i,"]"," = ",metrics.accuracy_score(TX,TY))

    AccuLIST.append( metrics.accuracy_score(TX,TY))



MeanAccu = np.mean(AccuLIST)
print("Mean Accuracy Across ", N, "Iterations = ", MeanAccu, "%" )
Rindex = AccuLIST.index(max(AccuLIST))
Rstate = random.seed(Rindex)
print("Max accuracy =",max(AccuLIST)," at AccuracyLIST[",Rindex+1,"]")# for loop starts at '1' !!


### MORE TREE PLOTS at Depth0 should match seed with higest accuracy
dtree = DecisionTreeClassifier(max_depth=Depth0, random_state=Rindex)
dtree = dtree.fit(X.loc[TraininSet], y.loc[TraininSet])
fig1 = plt.figure(figsize=(15.0,15.0))
tree.plot_tree(dtree, feature_names=features, filled=True, class_names=['No','Yes'])# tested using ['0','1']
plt.show()




















