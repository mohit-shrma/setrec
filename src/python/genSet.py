import sys
import random


def getUserItemsNMap(ratFileName):
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
      if user not in userMap:
        userMap[user] = u
        u += 1
      if item not in itemMap:
        itemMap[item] = i
        i+= 1
      rat = float(cols[2])
      if user not in userItemsRat:
        userItemsRat[user] = {}
      userItemsRat[user][item] = rat
  return (userItemsRat, userMap, itemMap)


def getSetsForUser(itemRats, nSetsPerUser, setSize, thresh):
  nItems = len(itemRats)
  setLabels = []

  if nItems < 1.5 * setSize:
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

    setLabels.append((tempSet, label))

  return setLabels


def genSetsNWrite(userItemsRat, opFileName, setSize, thresh, nSetsPerUser, uMap,
    iMap):
  with open(opFileName, 'w') as g:
    u = 0
    for user, itemRats in userItemsRat.iteritems():
      setLabels = getSetsForUser(itemRats, nSetsPerUser, setSize, thresh)
      nSets = len(setLabels)
      setItems = set([])
      for (st, label) in setLabels:
        setItems = setItems | st
      g.write(str(uMap[user]) + ' ' + str(nSets) + ' ' + str(len(setItems)) + '\n')
      for (st, label) in setLabels:
        stItm = map(lambda x: iMap[x], st)
        g.write(str(label) + ' ' + str(len(stItm)) + ' ' 
            + ' '.join(map(str, list(stItm))) + '\n')


def writeMap(m, opName):
  with open(opName, 'w') as g:
    for k, v in m.iteritems():
      g.write(str(k) + '\t' + str(v) + '\n')


def main():
  ratFileName = sys.argv[1]
  setSize = int(sys.argv[2])
  thresh = float(sys.argv[3])
  nSetsPerUser = int(sys.argv[4])
  opFileName = sys.argv[5]
  
  uMapFName = opFileName + '_u_map'
  iMapFName = opFileName + '_i_map'

  (userItemsRat, userMap, itemMap) = getUserItemsNMap(ratFileName)
  writeMap(userMap, uMapFName)
  writeMap(itemMap, iMapFName)
  genSetsNWrite(userItemsRat, opFileName, setSize, thresh, nSetsPerUser,
      userMap, itemMap)


if __name__ == '__main__':
  main()
