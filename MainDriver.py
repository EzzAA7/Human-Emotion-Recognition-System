import math
from scipy.io import wavfile
import matplotlib.pyplot as plt
import numpy as np
#import RandomizeFiles
from python_speech_features import *
from os import walk, getcwd, path, chdir
import joblib
# from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
# from sklearn.metrics import classification_report, confusion_matrix
# from sklearn.ensemble import RandomForestClassifier
# from sklearn.feature_selection import VarianceThreshold
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
# from sklearn import svm
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import scale
# from sklearn.cluster import KMeans
import pandas as pd
from sklearn.ensemble import ExtraTreesClassifier
from sklearn.linear_model import LogisticRegression
from mlxtend.classifier import StackingClassifier



# Feature Selection ( i did it here instead of the end of file because of memory issues)
def FeatureSelection():

    data = pd.read_csv("emot2.csv")
    X = data.iloc[:, 0:39]  # independent columns
    y = data.iloc[:, -1]  # target column i.e price range
    cols = list(data.head(0))
    model = ExtraTreesClassifier()
    model.fit(X, y)
    for i in range(len(cols) - 2):
        print(f'col: {cols[i]}:  {model.feature_importances_[i]}')  # use inbuilt class feature_importances of tree based classifiers
    # plot graph of feature importances for better visualization
    feat_importances = pd.Series(model.feature_importances_, index=X.columns)
    feat_importances.nsmallest(20).plot(kind='barh')
    plt.show()
    feat_importances.nlargest(20).plot(kind='barh')
    plt.show()
    feat_importances.nsmallest(5).plot(kind='barh')
    plt.show()
    feat_importances.nlargest(64).plot(kind='barh')
    plt.show()
    del (model, X, y, feat_importances)


# At the start we needed to create a method to randmize and rename files to 75% of them being Training and testing for
# the rest
# This is commented since we no longer needed it once we got our testing data set

#RandomizeFiles.Randomize75()

# Initialization for Training
mypath = getcwd()
og = getcwd()
# mypath = path.join(mypath, "Training75\\")
mypath = path.join(mypath, "Training100\\")
#mypath = path.join(mypath, "train/")
chdir(mypath)
dictOfClasses = {1: "anger", 2: "bordom", 3: "fear", 4: "happy", 5: "sadness", 6: "neutral"}

windowSize = 0.02  # 20 ms
shift = 0.01  # 10ms => 50%
numOfFilters = 20  # triangle filters


# We trained the first file to try our initial setup before appending to all files
def TrainFirstFile():
    global rate, data, mfcc_feat, fbankLogEnergy_feat, ssc_feat, Allfeatures, numberOfFrames, AllLabels
    rate, data = wavfile.read("anger1.wav")
    # maybe we have to subtract the DC
    # data=data-np.mean(data)
    mfcc_feat = mfcc(data, rate, winlen=windowSize, winstep=shift, winfunc=np.hamming, appendEnergy=True, numcep=numOfFilters)
    # delta_mfcc_feat = delta(mfcc_feat, 2)
    # ddelta_feat = delta(delta_mfcc_feat, 2)
    fbankLogEnergy_feat = logfbank(data, rate, winlen=windowSize, winstep=shift, nfilt=numOfFilters)
    ssc_feat = ssc(data, rate, winlen=windowSize, winstep=shift, nfilt=numOfFilters)
    # concatination all features
    Allfeatures = np.concatenate((mfcc_feat, fbankLogEnergy_feat, ssc_feat), axis=1)
    # Allfeatures = np.concatenate((mfcc_feat, delta_mfcc_feat), axis=1)
    # Allfeatures=fbankLogEnergy_feat
    # print(rate)   #Fs
    # print(data)   #smples
    # print(len(data))
    print("feattures shape : ", np.shape(Allfeatures))
    numberOfFrames = len(Allfeatures)
    print("number of frames : ", numberOfFrames)
    AllLabels = np.repeat(1, numberOfFrames)


TrainFirstFile()


# Train data: extract features and appended to the matrix of features and matrix of labels
def TrainData():
    global i, f, dirpath, dirnames, filenames, rate, data, mfcc_feat, fbankLogEnergy_feat, ssc_feat, numberOfFrames, Allfeatures, AllLabels
    for i in dictOfClasses:
        subPath = mypath + dictOfClasses[i]

        f = []
        for (dirpath, dirnames, filenames) in walk(subPath):
            f.extend(filenames)
            break
        print(len(filenames))

        for name in filenames:
            if (name == ".DS_Store" or name == "results.txt"):
                continue
            else:
                # print(getcwd())
                chdir(subPath)
                # print(getcwd())
                rate, data = wavfile.read(name)
                print(name)
                # maybe we have to subtract the DC
                # data = data - np.mean(data)
                mfcc_feat = mfcc(data, rate, winlen=windowSize, winstep=shift, winfunc=np.hamming, appendEnergy=True, numcep=numOfFilters)
                # delta_mfcc_feat = delta(mfcc_feat, 2) # Removed After FS
                # ddelta_feat = delta(delta_mfcc_feat, 2)   # Removed after FS
                fbankLogEnergy_feat = logfbank(data, rate, winlen=windowSize, winstep=shift,
                                               nfilt=numOfFilters)  # may be it would be 26-22
                ssc_feat = ssc(data, rate, winlen=windowSize, winstep=shift, nfilt=numOfFilters)
                # concatination all features
                features = np.concatenate((mfcc_feat, fbankLogEnergy_feat, ssc_feat), axis=1)
                # features = np.concatenate((mfcc_feat, delta_mfcc_feat), axis=1)
                # features=fbankLogEnergy_feat

                print("feattures shape : ", np.shape(features))
                numberOfFrames = len(features)
                print("number of frames : ", numberOfFrames)
                labels = np.repeat(i, numberOfFrames)  # to make an array as labels for the frames
                Allfeatures = np.concatenate((Allfeatures, features))
                print("Allfeattures shape : ", np.shape(Allfeatures))
                AllLabels = np.concatenate((AllLabels, labels), axis=0)
        print("All Labels length : ", len(AllLabels))
        print(AllLabels)
        print("******************************************************************************************")
        print("******************************************************************************************")
        print("******************************************************************************************")


TrainData()

# Function to save the model as a CSV for easier demonstration and preparing for Feature Selection
def SaveCSVandJOB():
    # Save as CSV for future use
    chdir(og)
    # # together=np.concatenate((Allfeatures, AllLabels))
    numpy.savetxt("emotion.csv", Allfeatures, delimiter=",")
    numpy.savetxt("emotion2.csv", AllLabels, delimiter=",")
    X_name = 'X.joblib'
    y_name = 'y.joblib'

    # change directory
    chdir(og)
    # testpath = path.join(og, "test25/")  # Old directory when it was 25% Testing
    testpath = path.join(og, "testNew/")  # New directory when Dr.Hanani sent us the new tests
    chdir(testpath)

    # Save in memory
    savedX = joblib.dump(Allfeatures, path.join(testpath, X_name))
    savedy = joblib.dump(AllLabels, path.join(testpath, y_name))


#The following two methods create
def createDecisionTree():
    global model1
    dtree = DecisionTreeClassifier()  # Regular Decision Tree
    dtree.fit(Allfeatures, AllLabels)
    model1 = dtree

def improvedRandomStateTree():
    global model
    dtree = DecisionTreeClassifier(criterion="entropy", random_state=50)  # Regular Decision Tree
    dtree.fit(Allfeatures, AllLabels)
    model = dtree


# training by decision tree
#createDecisionTree()    # This was sometimes used as model in testing
improvedRandomStateTree()   # This was  used as model in testing


def createGNBModel():
    global model
    gnb = GaussianNB(priors=[.2, .1, .15, .3, .2, .05])
    gnb.fit(Allfeatures, AllLabels)
    model = gnb

# This was used as a model once, its results can be seen exp3 in the report
#createGNBModel()

def createMLPCModel():
    global model
    model = MLPClassifier(alpha=0.01, batch_size=256, hidden_layer_sizes=(300,), learning_rate='adaptive', max_iter=500)
    model.fit(Allfeatures, AllLabels)

# This was used as a model once, its results can be seen exp2 in the report
#createMLPCModel()


# change the path to testing data
# mypath = path.join(og, "test25")    # Old directory when it was 25% Testing
mypath = path.join(og, "testNew")    # New directory when Dr.Hanani sent us the new tests
# mypath = path.join(og, "development")
chdir(mypath)
choice = 1


def TestData():
    global dicOfTrue, truePredictions, dicOfFalse, falsePrediction, dirpath, dirnames, filenames, i, rate, data, mfcc_feat, fbankLogEnergy_feat, ssc_feat
    resultsFile = open("results.txt", "w")
    dicOfTrue = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    truePredictions = 0
    dicOfFalse = {1: 0, 2: 0, 3: 0, 4: 0, 5: 0, 6: 0}
    falsePrediction = 0
    filesTest = []
    for (dirpath, dirnames, filenames) in walk(mypath):
        f.extend(filenames)
        break
    print(len(filenames))
    # tempArr= np.zeros()
    for file in filenames:
        if (file == ".DS_Store" or file == "results.txt"):
            continue
        else:
            realLabel = 0
            for i in dictOfClasses:  # loop to determine the true label in order to calculate the accuracy
                if (dictOfClasses[i] in file):
                    realLabel = i
                    break
            print(file)
            # choice = input("enter the file name to be tested or 0 to exit")
            rate, data = wavfile.read(file)
            # maybe we have to subtract the DC
            # data = data - np.mean(data)
            mfcc_feat = mfcc(data, rate, winlen=windowSize, winstep=shift, winfunc=np.hamming, appendEnergy=True, numcep=numOfFilters)
            # delta_mfcc_feat = delta(mfcc_feat, 1)
            # ddelta_feat = delta(delta_mfcc_feat, 1)
            fbankLogEnergy_feat = logfbank(data, rate, winlen=windowSize, winstep=shift, nfilt=numOfFilters)
            ssc_feat = ssc(data, rate, winlen=windowSize, winstep=shift, nfilt=numOfFilters)
            # concatination all features
            Test_feat = np.concatenate((mfcc_feat, fbankLogEnergy_feat, ssc_feat), axis=1)
            # Test_feat = np.concatenate((mfcc_feat, delta_mfcc_feat), axis=1)
            # Test_feat=fbankLogEnergy_feat
            # numberOfFrames = len(Test_feat)
            # Test_label = np.repeat(realLabel, numberOfFrames)

            predictions = model.predict(Test_feat)
            predictedLabel = np.bincount(predictions).argmax()
            print(file, "predicted as : ", predictedLabel)

            # printing the results
            resultsFile.writelines(
                str(file) + " labeled as : " + str(predictedLabel) + " it's real label is : " + str(realLabel) + "\n")

            # to calculating the accuracy
            if (predictedLabel == realLabel):
                dicOfTrue[realLabel] += 1
                truePredictions += 1
            else:
                dicOfFalse[realLabel] += 1
                falsePrediction += 1


TestData()


def PrintAccuracy():
    print("df")
    print("true = ", truePredictions)
    print("false = ", falsePrediction)
    for j in dictOfClasses:
        print(dictOfClasses[j], " accuracy = ", 100 * dicOfTrue[j] / (dicOfTrue[j] + dicOfFalse[j]), "%")
    print("Avg accuracy = ", 100 * truePredictions / (truePredictions + falsePrediction), "%")


PrintAccuracy()


##########################################################################################################
# THE FOLLOWING WERE TRIED BUT NOT USED IN THE FINAL PROJECT
#####################################################################################

def TestSpecificFile():
    global choice, rate, data, mfcc_feat, fbankLogEnergy_feat, ssc_feat, numberOfFrames
    while (choice != 0):
        choice = input("enter the file name to be tested or 0 to exit")
        rate, data = wavfile.read(choice)
        # maybe we have to subtract the DC
        data = data - np.mean(data)
        mfcc_feat = mfcc(data, rate, winlen=windowSize, winstep=shift, winfunc=np.hamming, appendEnergy=True)
        # delta_mfcc_feat = delta(mfcc_feat, 1)
        # ddelta_feat = delta(delta_mfcc_feat, 1)
        fbankLogEnergy_feat = logfbank(data, rate, winlen=windowSize, winstep=shift, nfilt=numOfFilters)
        ssc_feat = ssc(data, rate, winlen=windowSize, winstep=shift, nfilt=numOfFilters)
        # concatination all features
        Test_feat = np.concatenate((mfcc_feat, fbankLogEnergy_feat, ssc_feat), axis=1)
        numberOfFrames = len(Test_feat)
        Test_label = np.repeat(1, numberOfFrames)
        predictions = model.predict(Test_feat)
        print("#################################")
        print()
        # print(classification_report(Test_label, predictions))
        # print("the most frequent label is :", most_frequent(predictions))
        print("the most frequent label is : ", np.bincount(predictions).argmax())
        print("accuracy = ", accuracy_score(Test_label, predictions))
        print()

# TestSpecificFile()

# X_train, X_test, y_train, y_test = train_test_split(Allfeatures, AllLabels, test_size=0.30, random_state=42)

# sel = VarianceThreshold(threshold=(.9 * (1 - .9)))
# sel.fit_transform(X_train)
# sel.fit_transform(X_test)

# clf = MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(15,), random_state=1)
# clf.fit(Allfeatures,AllLabels)
# model=clf

# clf = svm.SVC(decision_function_shape='ovo')
# clf.fit(X_train, y_train)
# model=clf

# KMEANS
# x= scale(Allfeatures)   #Center to the mean and component wise scale to unit variance.
# y= pd.DataFrame(AllLabels)
# var_names = dictOfClasses
# km = KMeans(n_clusters=6)   # Started kmeans model
# km.fit(x)                   # fit data to it
#
# print(x[:,5])
# plt.scatter(x[:,0], x[:,5])
#
# clusters=km.cluster_centers_        #save centroids to make scatter easier
# y_km=km.fit_predict(x)
#
# plt.scatter(x[y_km==0,0], x[y_km==0,1], s=50, color='red')
# plt.scatter(x[y_km==1,0], x[y_km==1,1], s=50, color='green')
# plt.scatter(x[y_km==2,0], x[y_km==2,1], s=50, color='yellow')
# plt.scatter(x[y_km==3,0], x[y_km==3,1], s=50, color='cyan')
# plt.scatter(x[y_km==4,0], x[y_km==4,1], s=50, color='blue')
# plt.scatter(x[y_km==5,0], x[y_km==5,1], s=50, color='gray')
#
# plt.scatter(clusters[0][0], clusters[0][1], marker='*', s=200, color='black')
# plt.scatter(clusters[1][0], clusters[1][1], marker='*', s=200, color='black')
# plt.scatter(clusters[2][0], clusters[2][1], marker='*', s=200, color='black')
# plt.scatter(clusters[3][0], clusters[3][1], marker='*', s=200, color='black')
# plt.scatter(clusters[4][0], clusters[4][1], marker='*', s=200, color='black')
# plt.scatter(clusters[5][0], clusters[5][1], marker='*', s=200, color='black')
#
# plt.show()
# print(km.labels_)
