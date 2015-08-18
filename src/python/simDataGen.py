import sys
import numpy as np


def genLatFac(nUsers, nItems, facDim):
  uFac = np.random.normal(loc=0.5, scale=0.25, size=(nUsers, facDim))
  iFac = np.random.normal(loc=0.5, scale=0.25, size=(nUsers, facDim)) 
  return (uFac, iFac)


def saveMat(mat, opFileName):
  np.savetxt(opFileName, mat, fmt="%f")


def createData(nUsers, nItems, uFac, iFac, opFileName, density=0.1):
  
  userItemsRat = {}
  nRats = nUsers * nItems * density
  
  #for each user write ratings for 20 items
  for u in range(nUsers):
    userItemsRat[u] = {}
    while (len(userItemsRat[u]) < 20):
      item = np.random.randint(0, nItems)
      if item not in userItemsRat[u]:
        userItemsRat[u][item] = np.dot(uFac[u], iFac[item])
      
  #for each item make sure there are 20 users
  for item in range(nItems):
    for j in range(20):
      u = np.random.randint(0, nUsers)
      if item not in userItemsRat[u]:
        userItemsRat[u][item] = np.dot(uFac[u], iFac[item])
  
  nRatsTillNow = 0
  for u, itemRats in userItemsRat.iteritems():
    nRatsTillNow += len(itemRats)
  
  remRat = nRats - nRatsTillNow

  #get the remaining ratings based on specified density
  ctRat = 0
  while ctRat < remRat:
    u = np.random.randint(0, nUsers)
    item = np.random.randint(0, nItems)
    if item not in userItemsRat[u]:
      userItemsRat[u][item] = np.dot(uFac[u], iFac[item])
      ctRat += 1
  
  with open(opFileName, "w") as g:
    for u in range(nUsers):
      items = userItemsRat[u].keys()
      items.sort()
      for item in items:
        g.write(str(u) + ',' + str(item) + ',' + str(userItemsRat[u][item]) +
        '\n')



def main():
  nUsers = int(sys.argv[1])
  nItems = int(sys.argv[2])
  facDim = int(sys.argv[3])
  opPrefix = sys.argv[4]
  (uFac, iFac) = genLatFac(nUsers, nItems, facDim)
  saveMat(uFac, opPrefix + "_" + str(nUsers) + "_uFac.txt")
  saveMat(iFac, opPrefix + "_" + str(nItems) + "_iFac.txt")
  createData(nUsers, nItems, uFac, iFac, opPrefix + "_" + str(nUsers) + "_" +
      str(nItems) + "_ratings.csv")


if __name__ == '__main__':
  main()

