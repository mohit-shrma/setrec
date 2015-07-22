import sys
import random
import numpy as np
import heapq


def writeMap(m, opName):
  with open(opName, 'w') as g:
    for k, v in m.iteritems():
      g.write(str(k) + '\t' + str(v) + '\n')


def getEvalSet(userItemsRat, sampPerUser=1):
  evalSet = []
  users = userItemsRat.keys()
  
  for u in users:
    
    itemRats = userItemsRat[u]
    nUItems = len(itemRats)
    randInds = set([])
    
    while len(randInds) != sampPerUser:
      ind = random.randint(0, nUItems-1)
      randInds.add(ind)
    
    uItems = itemRats.keys()
    
    for ind in randInds:
      evalSet.append((u, uItems[ind], itemRats[uItems[ind]]))
    
    for ind in randInds:
      del itemRats[uItems[ind]]

  return evalSet


def getUserItemsNMap(ratFileName, setSize):
  userItemsRat = {}
  userMap = {}
  itemMap = {}
  u = 0
  i = 0 
  
  with open(ratFileName, 'r') as f:
    for line in f:
      cols = line.strip().split(',')
      user = int(cols[0])
      item = int(cols[1])
      rat = float(cols[2])
      if user not in userItemsRat:
        userItemsRat[user] = {}
      userItemsRat[user][item] = rat
  
  userKeys = userItemsRat.keys()
  for user in userKeys:
    itemRats = userItemsRat[user]
    #TODO: hard coded 
    if len(itemRats) < 20:
      #delete key if not sufficient item
      del userItemsRat[user]
      continue
    if user not in userMap:
      userMap[user] = u
      u += 1
    for item in itemRats.keys():
      if item not in itemMap:
        itemMap[item] = i
        i += 1

  return (userItemsRat, userMap, itemMap)


def writeEvalSet(evalSet, opFileName, uMap, iMap):
  with open(opFileName, 'w') as g:
    for s in evalSet:
      g.write(str(uMap[s[0]]) + ' ' + str(iMap[s[1]]) 
          + ' ' + str(s[2]) + '\n')


def getTriplets(userItemsRat):
  rats = []
  users = userItemsRat.keys()
  users.sort()
  for u in users:
    itemRats = userItemsRat[u]
    items = itemRats.keys()
    items.sort()
    for item in items:
      rats.append((u, item, userItemsRat[u][item]))
  return rats


def writeTriplets(userItemsRat, opFileName, uMap, iMap):
  rats = []
  users = userItemsRat.keys()
  users.sort()
  with open(opFileName, 'w') as g:
    for u in users:
      itemRats = userItemsRat[u]
      items = itemRats.keys()
      items.sort()
      for item in items:
        g.write(str(uMap[u]) + ' ' + str(iMap[item]) + ' ' +
            str(userItemsRat[u][item]) + '\n')
  return rats


def readMap(ipName):
  iMap = {}
  revIMap = {}
  with open(ipName, 'r') as f:
    for line in f:
      cols = line.strip().split()
      iMap[int(cols[0])] = int(cols[1])
      revIMap[int(cols[1])] = int(cols[0])
  return (iMap, revIMap)


def getSimMat(iFeatures):
  nItems = iFeatures.shape[0]
  sims = np.zeros((nItems, nItems))
  for i in range(nItems):
    for j in range(i+1, nItems):
      sims[i][j] = getSim(iFeatures[i], iFeatures[j])
      sims[j][i] = sims[i][j]
  return sims


def getSim(fVec1, fVec2):
  norm1 = np.linalg.norm(fVec1)
  norm2 = np.linalg.norm(fVec2)
  sim = np.dot(fVec1, fVec2)
  if norm1 > 0 and norm2 > 0:
    return sim/(norm1*norm2)
  else:
    return 0


#get the n most similar items to seed item
def getSimItems(seedItmInd, iFeatures, userItems, n, movDic, simMat):
  nRetItems = []
  simItems = []
  for i in range(len(userItems)):
    if userItems[i] != seedItmInd:
      simItems.append((simMat[movDic[seedItmInd]][movDic[userItems[i]]], 
        userItems[i]))
  simItems.sort(reverse = True)  
  nSimItems = heapq.nlargest(n, simItems)
  for (sim, item) in nSimItems:
    nRetItems.append(item)
  nRetItems.sort()
  return nRetItems


def getSimSetsForUser(itemRats, iFeatures, nSetsPerUser, setSize, movDic, 
    simMat):
  nItems = len(itemRats)
  setLabels = set([])
  
  if nItems < 2*setSize:
    return setLabels
  
  items = itemRats.keys()
  
  j = 0
  while len(setLabels) < nSetsPerUser and j < 5*nSetsPerUser:
    seedItmInd = random.randint(0, len(items)-1)
    tempSet = getSimItems(items[seedItmInd], iFeatures, items, setSize, movDic,
        simMat) 
    #assign set label
    setItemRat = [] 
    for item in tempSet:
      setItemRat.append((itemRats[item], item))
    setItemRat.sort()
    
    #get avg of top items
    sm = 0.0
    nTopItems = 0.0
    for i in range(setSize/2, setSize):
      sm += setItemRat[i][0]
      nTopItems += 1
    avgTopRatItems = sm/nTopItems
    
    tempList = map(str, list(tempSet))
    setLabels.add(( ' '.join(tempList), int(avgTopRatItems)))
    print 'j: ', j 
    j += 1 
  
  if j == 5*nSetsPerUser:
    print 'Failed to generate unique sets'
  
  setLabels = list(setLabels)
  for i in range(len(setLabels)):
    setLabels[i] = (set(map(int, setLabels[i][0].split(' '))), setLabels[i][1]) 
  return setLabels


def genSetsNWrite(userItemsRat, opFileName, setSize, nSetsPerUser,
    uMap, iMap, movDic, iFeatures, simMat):
  
  with open(opFileName, 'w') as g:
    u = 0
    for user, itemRats in userItemsRat.iteritems():
      
      setLabels = getSimSetsForUser(itemRats, iFeatures, nSetsPerUser, setSize,
          movDic, simMat)
      
      nSets = len(setLabels)
      if nSets == 0:
        print 'No set found'
      if (nSets < nSetsPerUser):
        print 'less sets found: ', user, nSets, nSetsPerUser, len(itemRats)
      
      setItems = set([])
      for (st, label) in setLabels:
        setItems = setItems | st
      setAftrMap = map(lambda x: iMap[x], setItems)
      
      print 'u : ', u

      g.write(str(uMap[user]) + ' ' + str(nSets) + ' ' + str(len(setAftrMap)) +
          ' ' + ' '.join(map(str, list(setAftrMap))) + '\n')
      for (st, label) in setLabels:
        stItm = map(lambda x: iMap[x], st)
        g.write(str(label) + ' ' + str(len(stItm)) + ' ' 
            + ' '.join(map(str, list(stItm))) + '\n')
      u += 1

def getFeatures(ipName, nFeatures = 20):
  nItems = 0
  with open(ipName, 'r') as f:
    for line in f:
      nItems += 1

  print "nItems: ", nItems

  iFeatureArr = np.zeros((nItems, nFeatures))

  with open(ipName, 'r') as f:
    i = 0
    for line in f:
      cols = line.strip().split()
      cols = map(int, cols)
      for ind in cols:
        iFeatureArr[i][ind] = 1.0
      i += 1 
  return iFeatureArr


def readMovieIds(ipName):
  movIds = []
  with open(ipName, 'r') as f:
    f.readline()
    for line in f:
      cols = line.strip().split(',')
      movIds.append(int(cols[0]))
  return movIds


def main():

  featName     = sys.argv[1]
  ratFileName  = sys.argv[2]
  movFName     = sys.argv[3]
  setSize      = int(sys.argv[4])
  nSetsPerUser = int(sys.argv[5])
  seed         = int(sys.argv[6])
  opFileName   = sys.argv[7]
  
  random.seed(seed)
  
  uMapFName = opFileName + '_u_map'
  iMapFName = opFileName + '_i_map'
  
  (userItemsRat, userMap, itemMap) = getUserItemsNMap(ratFileName, setSize)
  writeMap(userMap, uMapFName)
  writeMap(itemMap, iMapFName)
 
  #TODO: hardcoded no. of genres
  iFeatureArr     = getFeatures(featName, 20)
  movIds          = readMovieIds(movFName)
  
  movDic = dict(zip(movIds, range(len(movIds))))

  simMat  = getSimMat(iFeatureArr) 

  #get test set
  testSet = getEvalSet(userItemsRat)  
  writeEvalSet(testSet, opFileName + '_test', userMap, itemMap)

  #get validation set
  valSet = getEvalSet(userItemsRat)
  writeEvalSet(valSet, opFileName + '_val', userMap, itemMap)

  #write train set
  trainSet = getTriplets(userItemsRat)
  writeTriplets(userItemsRat, opFileName + '_train', userMap, itemMap)
  
  genSetsNWrite(userItemsRat, opFileName, setSize, nSetsPerUser,
    userMap, itemMap, movDic, iFeatureArr, simMat)



if __name__ == '__main__':
  main()


