import sys
import random


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


def getSetsForUser2(itemRats, nSetsPerUser, setSize, thresh):
  nItems = len(itemRats)
  setLabels = set([])

  if nItems < 2 * setSize:
    return setLabels
  
  items = itemRats.keys()

  while len(setLabels) < nSetsPerUser:
    #generate set i
    tempSet = set([])
    while len(tempSet) < setSize:
      item = items[random.randint(0, nItems-1)]
      tempSet.add(item)
    
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
   
    """
    label = 1.0
    if avgTopRatItems < thresh:
      label = -1.0
    """

    tempList = map(str, list(tempSet))
    #TODO: ensure unique sets
    setLabels.add(( ' '.join(tempList), int(avgTopRatItems)))

  setLabels = list(setLabels)
  for i in range(len(setLabels)):
    setLabels[i] = (set(map(int, setLabels[i][0].split(' '))), setLabels[i][1]) 
  return setLabels


def getSetsForUser(itemRats, nSetsPerUser, setSize, thresh):
  nItems = len(itemRats)
  setLabels = []

  if nItems < 2 * setSize:
    return setLabels
  
  items = itemRats.keys()

  for i in range(nSetsPerUser):
    #generate set i
    tempSet = set([])
    while len(tempSet) < setSize:
      item = items[random.randint(0, nItems-1)]
      tempSet.add(item)
    
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
    
    label = 1.0
    if avgTopRatItems < thresh:
      label = -1.0

    setLabels.append((tempSet, int(avgTopRatItems)))

  return setLabels


def genSetsNWrite(userItemsRat, opFileName, setSize, thresh, nSetsPerUser, uMap,
    iMap):
  with open(opFileName, 'w') as g:
    u = 0
    for user, itemRats in userItemsRat.iteritems():
      setLabels = getSetsForUser2(itemRats, nSetsPerUser, setSize, thresh)
      if len(setLabels) == 0:
        print 'no set found'
      nSets = len(setLabels)

      if (nSets < nSetsPerUser):
        print 'less sets found: ', user, nSets, nSetsPerUser, len(itemRats)

      setItems = set([])
      for (st, label) in setLabels:
        setItems = setItems | st
      
      setAftrMap = map(lambda x: iMap[x], setItems)

      g.write(str(uMap[user]) + ' ' + str(nSets) + ' ' + str(len(setAftrMap)) +
          ' ' + ' '.join(map(str, list(setAftrMap))) + '\n')
      for (st, label) in setLabels:
        stItm = map(lambda x: iMap[x], st)
        g.write(str(label) + ' ' + str(len(stItm)) + ' ' 
            + ' '.join(map(str, list(stItm))) + '\n')


def writeMap(m, opName):
  with open(opName, 'w') as g:
    for k, v in m.iteritems():
      g.write(str(k) + '\t' + str(v) + '\n')


def main():
  ratFileName  = sys.argv[1]
  setSize      = int(sys.argv[2])
  thresh       = float(sys.argv[3])
  nSetsPerUser = int(sys.argv[4])
  seed         = int(sys.argv[5])
  opFileName   = sys.argv[6]

  random.seed(seed)

  uMapFName = opFileName + '_u_map'
  iMapFName = opFileName + '_i_map'
  
  (userItemsRat, userMap, itemMap) = getUserItemsNMap(ratFileName, setSize)
  writeMap(userMap, uMapFName)
  writeMap(itemMap, iMapFName)
 
  #get test set
  testSet = getEvalSet(userItemsRat)  
  writeEvalSet(testSet, opFileName + '_test', userMap, itemMap)

  #get validation set
  valSet = getEvalSet(userItemsRat)
  writeEvalSet(valSet, opFileName + '_val', userMap, itemMap)

  #write train set
  trainSet = getTriplets(userItemsRat)
  writeTriplets(userItemsRat, opFileName + '_train', userMap, itemMap)
  #writeEvalSet(trainSet, opFileName + '_train', userMap, itemMap)

  genSetsNWrite(userItemsRat, opFileName, setSize, thresh, nSetsPerUser,
      userMap, itemMap)


if __name__ == '__main__':
  main()
