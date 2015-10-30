import sys
import random
import os

def getEvalSet(userItemsRat, sampPerUser=20):
  evalSet = []
  users = userItemsRat.keys()
  users.sort()
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
    #evalSet = sorted(evalSet, key=lambda x: x[1])
    for ind in randInds:
      del itemRats[uItems[ind]]
  return evalSet


def writeEvalSet(evalSet, opFileName, uMap, iMap):
  with open(opFileName, 'w') as g:
    for s in evalSet:
      g.write(str(uMap[s[0]]) + ' ' + str(iMap[s[1]]) 
          + ' ' + str(s[2]) + '\n')


def writeEvalSetToCSR(evalSet, opFileName, uMap, iMap):
 
  currRowInd = 0
  currRow = []
  
  with open(opFileName + '.csr', 'w') as g:
    for s in evalSet:
      rowInd = uMap[s[0]]
      colInd = iMap[s[1]]
      rat    = s[2]  
      if rowInd != currRowInd:
        #new row found, write out current row
        g.write(' '.join(map(str, currRow)) + '\n')
        currRowInd += 1
        while currRowInd < rowInd:
          g.write('\n')
          currRowInd += 1
        currRow = []
      if rat != 0:
        currRow.append(colInd)
        currRow.append(rat)
    #write out last row
    g.write(' '.join(map(str, currRow)) + '\n')


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


def writeTripletsTOCSR(userItemsRat, opFileName, uMap, iMap):
  rats = []
  users = userItemsRat.keys()
  users.sort()
  with open(opFileName + '.csr', 'w') as g:
    for u in users:
      itemRats = userItemsRat[u]
      items = itemRats.keys()
      items.sort()

      for item in items:
        g.write(str(iMap[item]) + ' ' + str(userItemsRat[u][item]) + ' ')
      g.write('\n')
  return rats


def getUserItemsNMap(ratFileName, setSize):
  userItemsRat = {}
  userMap = {}
  itemMap = {}
  u = 0
  i = 0 
  obsItemIds = set([])
  with open(ratFileName, 'r') as f:
    for line in f:
      cols = line.strip().split(',')
      user = int(cols[0])
      item = int(cols[1])
      rat = float(cols[2])
      obsItemIds.add(item)
      if user not in userItemsRat:
        userItemsRat[user] = {}
      userItemsRat[user][item] = rat
  
  userKeys = userItemsRat.keys()
  userKeys.sort()

  minItemId = min(list(obsItemIds))
  maxItemId = max(list(obsItemIds))
  if minItemId ==0 and maxItemId - minItemId + 1 == len(obsItemIds):
    #found all items in order
    print 'found all items in order'
    for itemId in list(obsItemIds):
      itemMap[itemId] = itemId
  else:
    print 'found items not in order'

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


def getSetsForUser(itemRats, nSetsPerUser, setSize):
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
    setItemRat.sort(reverse=True)
    
    #get avg of top items
    sm = 0.0
    majSz = setSize/2
    if setSize %2 != 0:
      majSz = setSize/2 + 1

    #majSz = setSize

    for i in range(majSz):
      sm += setItemRat[i][0]
    avgTopRatItems = sm/(majSz*1.0)
   
    #get items in decrease order by rating
    itemsInDecrOrder = map(lambda x: str(x[1]), setItemRat)

    #tempList = map(str, list(tempSet))
    #TODO: ensure unique sets
    setLabels.add((' '.join(itemsInDecrOrder), float(avgTopRatItems)))

  setLabels = list(setLabels)
  for i in range(len(setLabels)):
    setLabels[i] = (set(map(int, setLabels[i][0].split(' '))), setLabels[i][1]) 
  return setLabels


def getNonOverlapSetsForUser(itemRats, nSetsPerUser, setSize):
  nItems    = len(itemRats)
  setLabels = set([])
  uItems    = set([])

  if nItems < 2 * setSize:
    return setLabels
  
  items = itemRats.keys()
  
  while len(setLabels) < nSetsPerUser:
    #generate set i
    tempSet = set([])
    nTry = 0
    while len(tempSet) < setSize and nTry < 5000:
      item = items[random.randint(0, nItems-1)]
      if item not in uItems:
        tempSet.add(item)
        uItems.add(item)
      nTry += 1

    if nTry >= 5000 and len(tempSet) < setSize:
      print 'Err: cant create sets with non overlapping items', len(items), len(setLabels)
      break

    #assign set label
    setItemRat = [] 
    for item in tempSet:
      setItemRat.append((itemRats[item], item))
    setItemRat.sort(reverse=True)
    
    #get avg of top items
    sm = 0.0
    majSz = setSize/2
    if setSize %2 != 0:
      majSz = setSize/2 + 1

    #majSz = setSize

    for i in range(majSz):
      sm += setItemRat[i][0]
    avgTopRatItems = sm/(majSz*1.0)
   
    #get items in decrease order by rating
    itemsInDecrOrder = map(lambda x: str(x[1]), setItemRat)

    #tempList = map(str, list(tempSet))
    #TODO: ensure unique sets
    setLabels.add((' '.join(itemsInDecrOrder), float(avgTopRatItems)))

  setLabels = list(setLabels)
  for i in range(len(setLabels)):
    setLabels[i] = (set(map(int, setLabels[i][0].split(' '))), setLabels[i][1]) 
  return setLabels


def genSetsNWrite(userItemsRat, opFileName, setSize, nSetsPerUser, uMap,
    iMap):
  uLess = []
  with open(opFileName, 'w') as g:
    u = 0
    for user, itemRats in userItemsRat.iteritems():
      setLabels = getSetsForUser(itemRats, nSetsPerUser, setSize)
      if len(setLabels) == 0:
        print 'no set found'
      nSets = len(setLabels)

      if (nSets < nSetsPerUser):
        print 'less sets found: ', user, nSets, nSetsPerUser, len(itemRats)
        uLess.append(user)

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
  print 'nUsers with less set:', len(uLess)  


def writeMap(m, opName):
  with open(opName, 'w') as g:
    for k, v in m.iteritems():
      g.write(str(k) + '\t' + str(v) + '\n')


def main():
  ratFileName  = sys.argv[1]
  setSize      = int(sys.argv[2])
  nSetsPerUser = int(sys.argv[3])
  seed         = int(sys.argv[4])
  opPrefix     = sys.argv[5]

  opPrefix     = opPrefix + '_' + str(setSize) + '_' + str(nSetsPerUser) \
                  + '_' + str(seed)

  #create op dir in current dir
  opDir = os.path.join(os.getcwd(), opPrefix)
  os.mkdir(opDir)

  random.seed(seed)

  uMapFName = os.path.join(opDir, opPrefix + '_u_map')
  iMapFName = os.path.join(opDir, opPrefix + '_i_map')
  
  (userItemsRat, userMap, itemMap) = getUserItemsNMap(ratFileName, setSize)
  writeMap(userMap, uMapFName)
  writeMap(itemMap, iMapFName)
 
  #get test set
  testSet = getEvalSet(userItemsRat)
  testFName = os.path.join(opDir, opPrefix + '_test')
  writeEvalSet(testSet, testFName, userMap, itemMap)
  writeEvalSetToCSR(testSet, testFName, userMap, itemMap)

  #get validation set
  valSet = getEvalSet(userItemsRat)
  valFName = os.path.join(opDir, opPrefix + '_val')
  writeEvalSet(valSet, valFName, userMap, itemMap)
  writeEvalSetToCSR(valSet, valFName, userMap, itemMap)

  #write train set
  trainSet = getTriplets(userItemsRat)
  trainFName = os.path.join(opDir, opPrefix + '_train')
  writeTriplets(userItemsRat, trainFName, userMap, itemMap)
  writeTripletsTOCSR(userItemsRat, trainFName, userMap, itemMap)
  #writeEvalSet(trainSet, opFileName + '_train', userMap, itemMap)

  setFName = os.path.join(opDir, opPrefix + '_sets')
  genSetsNWrite(userItemsRat, setFName, setSize, nSetsPerUser,
      userMap, itemMap)


if __name__ == '__main__':
  main()

