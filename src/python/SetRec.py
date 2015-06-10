import sys
import random
from scipy.stats import pearsonr 
from sklearn.metrics import mean_squared_error
import numpy as np

from  Model import Model
from ModelSimWt import ModelSimWt
from UserSets import UserSets

def loadData(ipFileName, nUsers):
  arrUserSets = []
  
  with open(ipFileName, 'r') as f:
    for u in range(nUsers):
      line = f.readline()
      cols = line.strip().split()
      
      user       = int(cols[0])
      numSets    = int(cols[1])
      nUserItems = int(cols[2])
      items      = map(int, cols[3:])
      items.sort()
      userSets = UserSets(user, numSets, items)
      
      for i in range(numSets):
        line = f.readline()
        cols = line.strip().split()
        label = int(cols[0])
        setSz = int(cols[1])
        itemSet = map(int, cols[2:]) 
        userSets.addSet(itemSet, label)
      userSets.initWt()
      arrUserSets.append(userSets)
  return arrUserSets


def writeData(opFileName, arrUserSets):
  with open(opFileName, 'w') as g:
    for userSet in arrUserSets:
      l = [userSet.user, userSet.numSets, len(userSet.items)]
      l += userSet.items
      l = map(str, l)
      g.write(' '.join(l) + '\n')
      for setInd in range(userSet.numSets):
        l = [userSet.labels[setInd], len(userSet.itemSets[setInd])]
        l += userSet.itemSets[setInd]
        l = map(str, l)
        g.write(' '.join(l) + '\n')


def baselineRMSETestScores(arrUserSets):
  testLabels = []
  baseline1Scores = []
  baseline2Scores = []
  for userSet in arrUserSets:
    testLabels += userSet.testLabels()
    baseline1Scores += userSet.baseline1TestSetScore()
    baseline2Scores += userSet.baseline2TestSetScore()
  base1rmse = mean_squared_error(testLabels, baseline1Scores)**0.5
  base2rmse = mean_squared_error(testLabels, baseline2Scores)**0.5 
  return (base1rmse, base2rmse)


def baselineCorrTestScores(arrUserSets):
  testLabels = []
  baseline1Scores = []
  baseline2Scores = []
  for userSet in arrUserSets:
    testLabels += userSet.testLabels()
    baseline1Scores += userSet.baseline1TestSetScore()
    baseline2Scores += userSet.baseline2TestSetScore()
  base1Score = pearsonr(testLabels, baseline1Scores)
  base2Score = pearsonr(testLabels, baseline2Scores)
  return (base1Score, base2Score)


def main():

  ipFileName = sys.argv[1]
  nUsers = int(sys.argv[2])
  nItems = int(sys.argv[3])
  facDim = int(sys.argv[4])
  regU = float(sys.argv[5])
  regI = float(sys.argv[6])
  learnrate = float(sys.argv[7])
  useSim = bool(sys.argv[8])
  maxIter = int(sys.argv[9])
  seed = int(sys.argv[10])

  random.seed(seed)

  print 'Loading data...'

  arrUserSets = loadData(ipFileName, nUsers)
  
  print 'Computing baseline scores...'
  (baseline1Score, baseline2Score)  = baselineCorrTestScores(arrUserSets)
  print "First baseline corr score: ", baseline1Score
  print "Second baseline corr score: ", baseline2Score

  print 'Computing rmse scores'
  (base1rmse, base2rmse) = baselineRMSETestScores(arrUserSets)
  print 'First baseline rmse score: ', base1rmse
  print 'Second baseline rmse score: ', base2rmse

  #writeData("tempOp", arrUserSets)
  
  #model = Model(nUsers, nItems, facDim, regU, regI, learnrate, maxIter, useSim)
  
  #model.train(arrUserSets)

  modelWt = ModelSimWt(nUsers, nItems, facDim, regU, regI, learnrate, maxIter, useSim)
  for i in range(5):
    print 'gradient check for user', i
    modelWt.gradCheck(arrUserSets[i]) 


if __name__ == '__main__':
  main()






