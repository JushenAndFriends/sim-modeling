
#%% import
import os
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import re
#%%test


#%% def
def readData(fileDir):
    # first create empty features and labels for later filling
    features=np.empty((1,60))
    labels=[]
    for file in os.listdir(fileDir):
        #read files and get the features and label and append them to the output
        filePath=os.path.join(fileDir, file)
        feature=getFeature(filePath).reshape((-1,60))
        # samples are in rows, feature are in columns
        features=np.concatenate((features,feature),axis=0)
        label=getLabel(file)
        labels.append(label)
    # the first row of the feature are empty, drop it.
    return (features[1:,:],labels)


def getFeature(filePath):
    # read the csv file as a table
    data=pd.read_csv(filePath)
    # drop duplicated rows, as the time many have duplication and when doing interpolation this will cause problems
    data.drop_duplicates(inplace=True)
    # drop nan rows
    data.dropna(inplace=True)
    # fix the aspect ration
    data.loc[:,"X"]=data.loc[:,"X"]*1.5
    # center the trace
    data.loc[:,"X"]=data.loc[:,"X"]-data.loc[:,"X"].mean()
    data.loc[:,"Y"]=data.loc[:,"Y"]-data.loc[:,"Y"].mean()
    dataArray=data.to_numpy()
    # interpolation over Time axis, making a trace sampled at equal interval, and scale them to same length. 
    # so all letter has a trace length of 2*30. Every sample must have the same number of features.
    # Time is dropped as we tranformed the data as all the letters are written with same duration. 
    # But the speed variation in the written trace is kept, as we use a interpolation.
    newX=np.interp(np.linspace(dataArray[0,0],dataArray[-1,0],30),
        dataArray[:,0], data.loc[:,"X"])
    newY=np.interp(np.linspace(dataArray[0,0],dataArray[-1,0],30),
        dataArray[:,0], data.loc[:,"Y"])
    # 2*30->1*60
    feature=np.concatenate((newX,newY),axis=None)
    return feature


def getLabel(fileName):
    # using a regex to get the letter from the file name.
    matchstr='(.*?)_(.)_(.*?).txt'
    letter=re.match(matchstr,fileName).group(2)
    return letter

# the NN only accept numeric input, so the Y letters have to be presented in numbers
def letter2Number(letter):
    a2z=getAlphabet()
    return a2z.index(letter)             

def number2Letter(index):
    a2z=getAlphabet()
    return a2z[index]

def getAlphabet():
    alpha = 'A'
    test_list=[]
    for i in range(0, 26): 
        test_list.append(alpha) 
        alpha = chr(ord(alpha) + 1)
    return test_list


