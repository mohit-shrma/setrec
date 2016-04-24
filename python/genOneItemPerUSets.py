import sys
import random
import numpy as np

def getSetsForUser(userFac, iFac, nSetsPerUser, setSize):
  items = set([])
  nItems = iFac.shape[0]
  setLabels = []
  for i in range(nSetsPerUser):
    #generate set
    itemSet = set([])
    while len(itemSet) < setSize:
      item = random.randint(0, nItems-1)
      if item not in items:
        itemSet.add(item)
        items.add(item)
    #generate set rating
    setItemRats = []
    for item in list(itemSet):
      setItemRats.append((np.dot(userFac, iFac[item]), item))
    setItemRats.sort()
    #get avg of rats
    sm = 0.0
    for (rat, item) in setItemRats:
      sm += rat
    avgItemRats = sm/len(setItemRats)
    setLabels.append((itemSet, avgItemRats))
  return (setLabels, items)  
  

def genSetsNWrite(uFac, iFac, setSize, nSetsPerUser, opFileName):
  nUsers = uFac.shape[0]
  nItems = iFac.shape[0]
  uItems = {}
  with open(opFileName, 'w') as g:
    for u in range(nUsers):
      (setLabels, items) = getSetsForUser(uFac[u], iFac, nSetsPerUser, setSize)
      uItems[u] = items
      nSets = len(setLabels)
      g.write(str(u) + ' ' + str(nSets) + ' ' + str(len(items)) + ' ' 
          + ' '.join(map(str, list(items))) + '\n')
      for (st, label) in setLabels:
        g.write(str(label) + ' ' + str(len(st)) + ' ' 
            + ' '.join(map(str, list(st))) + '\n')
  return uItems        


def writeTriplets(uFac, iFac, uItems, nUsers, opFileName):
  with open(opFileName, 'w') as g:
    for u in range(nUsers):
      for item in uItems[u]:
        g.write(str(u) + ' ' + str(item) + ' ' + str(np.dot(uFac[u],
          iFac[item])) + '\n')


def getTestValItems(nUsers, nItems, uItems, testCt = 10, valCt = 10):
  uTestItems = {}
  uValItems  = {}
  for u in range(nUsers):
    testValItem = set([])
    while len(testValItem) < testCt + valCt:
      item = random.randint(0, nItems-1)
      if item not in uItems[u]:
        testValItem.add(item)
    tvItems = list(testValItem)
    uTestItems[u] = tvItems[:testCt]
    uValItems[u] = tvItems[testCt:]
  return (uTestItems, uValItems)
  

def main():
  uFacName     = sys.argv[1]
  iFacName     = sys.argv[2]
  setSize      = int(sys.argv[3])
  nSetsPerUser = int(sys.argv[4])
  seed         = int(sys.argv[5])
  opPrefix   = sys.argv[6]

  random.seed(seed)

  uFac = np.loadtxt(uFacName)
  iFac = np.loadtxt(iFacName)
  nUsers = uFac.shape[0]
  nItems = iFac.shape[0]

  print 'nUsers: ', nUsers
  print 'nItems: ', nItems
  print 'setSize: ', setSize
  print 'nSetsPerUser: ', nSetsPerUser

  #get sets n write
  uItems = genSetsNWrite(uFac, iFac, setSize, nSetsPerUser, opPrefix + '_sets')

  #write triplets using uItems
  writeTriplets(uFac, iFac, uItems, nUsers, opPrefix + '.triplets') 

  #get test val items
  (uTestItems, uValItems) = getTestValItems(nUsers, nItems, uItems)

  writeTriplets(uFac, iFac, uTestItems, nUsers, opPrefix + '.test.triplets')
  writeTriplets(uFac, iFac, uValItems, nUsers, opPrefix + '.val.triplets')
  

if __name__ == '__main__':
  main()


