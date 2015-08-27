import sys
import numpy as np
import random

def genTestItems(nUsers, nItems, userFac, itemFac, trainItems, valItems,
    nTestItems, opName):
  
  with open(opName, 'w') as g:
    for u in range(nUsers):
      userItems = set(trainItems[u])
      userItems = userItems | set(valItems[u])
      userTestItems = set([])
      while len(userTestItems) < nTestItems:
        testItem = random.randint(0, nItems-1)  
        if testItem in userItems or testItem in userTestItems:
          continue
        userTestItems.add(testItem)
      #write all testItems with rating for user
      for testItem in userTestItems:
        rating = np.dot(userFac[u], itemFac[testItem])
        g.write(str(testItem) + ' ' + str(rating) + ' ')
      g.write('\n')


def getItems(csrMatName):
  u = 0;
  itemD = {}
  with open(csrMatName, 'r') as f:
    for line in f:
      itemD[u] = []
      cols = line.strip().split()
      for i in range(0, len(cols), 2):
        itemD[u].append(int(cols[i]))
      u += 1
  return itemD

def main():
  
  trainMatName = sys.argv[1]
  valMatName   = sys.argv[2]
  uLatFacFName = sys.argv[3]
  iLatFacFName = sys.argv[4]
  nTestItems   = int(sys.argv[5])
  opName       = sys.argv[6]

  uFac       = np.loadtxt(uLatFacFName)
  iFac       = np.loadtxt(iLatFacFName)
  nUsers     = uFac.shape[0]
  nItems     = iFac.shape[0]
 
  print 'nUsers: ', nUsers
  print 'nItems: ', nItems
  
  trainItems = getItems(trainMatName)
  print 'users in train: ', len(trainItems)
  
  valItems   = getItems(valMatName)
  print 'users in validation: ', len(valItems)

  genTestItems(nUsers, nItems, uFac, iFac, trainItems, valItems, nTestItems,
      opName) 
  

if __name__ == '__main__':
  main()




