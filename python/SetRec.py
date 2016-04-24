import sys
import random
from scipy.stats import pearsonr 
from sklearn.metrics import mean_squared_error
import numpy as np

from Model import Model
from ModelBase import ModelBase
from ModelSimWt import ModelSimWt
from UserSets import UserSets


def computeJaccSim(arrUserSets, nItems):
  
  nSets = 0
  for userSet in arrUserSets:
    nSets += userSet.numSets - len(userSet.testSetInds) - len(userSet.valSetInds)
  
  itemSets = {}
  setTillNow = 0

  for userSet in arrUserSets:
    for setInd in range(len(userSet.itemSets)):
      if setInd not in userSet.testSetInds and setInd not in userSet.valSetInds:
        for item in userSet.itemSets[setInd]:
          if item not in itemSets:
            itemSets[item] = set([])
          itemSets[item].add(setTillNow)
        setTillNow += 1
  
  print 'setTillNow: ', setTillNow, 'nSets: ', nSets
  
  sim = np.zeros((nItems, nItems))  
  for i in range(nItems):
    if i%100 == 0:
      print str(i)+'...'
    for j in range(i+1, nItems):
      sim[i][j] = float(len(itemSets[i] & itemSets[j]))/float(len(itemSets[i] | itemSets[j])) 
      sim[j][i] = sim[i][j]
  
  return sim


def computeSideInfoSim(arrUserSets, featureMat):

  for userSet in arrUserSets:
    
    u = userSet.user
    
    for s in range(len(userSet.itemSets)):
      
      sim = 0.0
      itemSet = userSet.itemSets[s]
      
      if (len(itemSet) <= 1):
        continue

      nPairs = 0
      
      for i in range(len(itemSet)):
        normI = np.linalg.norm(featureMat[itemSet[i]])
        
        if (normI == 0):
          continue
        
        for j in range(i, len(itemSet)):
          normJ = np.linalg.norm(featureMat[itemSet[j]])
        
          if normJ == 0:
            continue
      
          nPairs += 1
          sim += np.dot(featureMat[itemSet[i]], featureMat[itemSet[j]])/(normI*normJ) 
     
      if (nPairs > 0):
        sim = sim/nPairs
      else:
        sim = 0

      print u, s, len(itemSet), sim


def computeItemSims(nItems, featureMat):
  for i in range(nItems):
    normI = np.linalg.norm(featureMat[i])
    if normI == 0:
      continue
    for j in range(i+1, nItems):
      normJ = np.linalg.norm(featureMat[j])
      if normJ == 0:
        continue
      print i, j, np.dot(featureMat[i], featureMat[j])/(normI*normJ)


def loadItemFeat(itemFeatFName, nItems, nFeatures):
  itemFeats = np.zeros((nItems, nFeatures))
  with open(itemFeatFName, 'r') as f:
    i = 0
    for line in f:
      cols = line.strip().split()
      for k in range(0, len(cols), 2):
        itemFeats[i][int(cols[k])] = float(cols[k+1])
      i += 1
  return itemFeats


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
 
  """
  TODO: to compute baseline scores first train like C code 
        then estimate it.
  """

  with open('base2score.txt', 'w') as g:
    for score in baseline2Scores:
      g.write('\n' + str(score));
  
  with open('testInds.txt', 'w') as g:
    for userSet in arrUserSets:
      g.write(str(userSet.testSetInds[0]) + '\n')

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

  ipFileName      = sys.argv[1]
  nUsers          = int(sys.argv[2])
  nItems          = int(sys.argv[3])
  facDim          = int(sys.argv[4])
  regU            = float(sys.argv[5])
  regI            = float(sys.argv[6])
  learnrate       = float(sys.argv[7])
  useSim          = bool(sys.argv[8])
  maxIter         = int(sys.argv[9])
  seed            = int(sys.argv[10])
  featureMatFName = sys.argv[11]

  random.seed(seed)

  #print 'Loading data...'

  arrUserSets = loadData(ipFileName, nUsers)
  #sim = computeJaccSim(arrUserSets, nItems) 
  
  featureMat = loadItemFeat(featureMatFName, nItems, 1623)
  #computeSideInfoSim(arrUserSets, featureMat)
  computeItemSims(nItems, featureMat)

  """
  with open('jacSim.txt', 'w') as g:
    for i in range(nItems):
      for j in range(i+1, nItems):
        g.write('\n' + str(i) + ' ' + str(j) + ' ' + str(sim[i][j]))
  """


  """
  print 'Computing baseline scores...'
  (baseline1Score, baseline2Score)  = baselineCorrTestScores(arrUserSets)
  print "First baseline corr score: ", baseline1Score
  print "Second baseline corr score: ", baseline2Score

  print 'Computing rmse scores'
  (base1rmse, base2rmse) = baselineRMSETestScores(arrUserSets)
  print 'First baseline rmse score: ', base1rmse
  print 'Second baseline rmse score: ', base2rmse
  """

  #writeData("tempOp", arrUserSets)
  
  #model = Model(nUsers, nItems, facDim, regU, regI, learnrate, maxIter, useSim)
  
  #model.train(arrUserSets)
  
  """
  modelWt = ModelSimWt(nUsers, nItems, facDim, regU, regI, learnrate, maxIter, useSim)
  for i in range(5):
    print 'gradient check for user', i
    modelWt.gradCheck(arrUserSets[i]) 
  """
  
  """
  modelBase = ModelBase(nUsers, nItems)
  modelBase.train(arrUserSets)

  print 'modelBase Validation Err', modelBase.valErr(arrUserSets)
  print 'modelBase Test Err', modelBase.testErr(arrUserSets)
  """

if __name__ == '__main__':
  main()






