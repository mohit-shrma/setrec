import numpy as np
import random


class ModelBase:

  def __init__(self, nUsers, nItems):
    self.nUsers = nUsers
    self.nItems = nItems


  def train(self, arrUserSets):
    for userSet in arrUserSets:
      user = userSet.user
      #got through items preferred by user
      for i in range(len(userSet.items)):
        item = userSet.items[i]
        userSet.itemsWt[i] = 0
        nTestValSet = 0
        #go through sets where item appears
        for setInd in userSet.item2Sets[item]:
          if setInd in userSet.testSetInds or setInd in userSet.valSetInds:
            nTestValSet += 1;
            continue
          userSet.itemsWt[i] += userSet.labels[setInd] / len(userSet.itemSets[setInd])

        if len(userSet.item2Sets[item]) > nTestValSet:
          userSet.itemsWt[i] = userSet.itemsWt[i]/(len(userSet.item2Sets[item]) - nTestValSet)
        

  def testErr(self, arrUserSets):
    
    testLabels = []
    testScores = []
    
    for userSet in arrUserSets:
      user = userSet.user
      itemWtDict = dict(zip(userSet.items, userSet.itemsWt))
      for setInd in userSet.testSetInds:
        testLabels.append(userSet.labels[setInd])
        estScore = 0
        for item in userSet.itemSets[setInd]:
          #get the wt of item from itemsWt array
          estScore += itemWtDict[item]
        testScores.append(estScore)

    with open('UserSet0Wt.txt', 'w') as g:
      itemWtDict = dict(zip(arrUserSets[0].items, arrUserSets[0].itemsWt))
      for item, wt in itemWtDict.iteritems():
        g.write(str(item) + ' ' + str(wt) + '\n')

    with open('testLabelsScores.txt', 'w') as g:
      for i in range(len(testLabels)):
        g.write(str(testLabels[i]) + '\t' + str(testScores[i]) + '\n')


    rmse = 0
    for i in range(len(testLabels)):
      rmse += (testLabels[i]-testScores[i])*(testLabels[i] - testScores[i])
    rmse = np.sqrt(rmse/len(testScores))
    
    return rmse
    

  def valErr(self, arrUserSets):
    
    valLabels = []
    valScores = []
    
    for userSet in arrUserSets:
      user = userSet.user
      itemWtDict = dict(zip(userSet.items, userSet.itemsWt))
      for setInd in userSet.valSetInds:
        valLabels.append(userSet.labels[setInd])
        estScore = 0
        for item in userSet.itemSets[setInd]:
          #get the wt of item from itemsWt array
          estScore += itemWtDict[item]
        valScores.append(estScore)
    
    rmse = 0
    for i in range(len(valLabels)):
      rmse += (valLabels[i]-valScores[i])*(valLabels[i] - valScores[i])
    rmse = np.sqrt(rmse/len(valScores))
    
    return rmse
    






