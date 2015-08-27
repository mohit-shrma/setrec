import sys
import numpy as np


def getHR(uFac, iFac, testItemsRat, N=10):
  
  nUsers = len(testItemsRat)
  avgHitRate = 0.0

  for u in range(nUsers):
    itemsRat = testItemsRat[u]
    ratItemOrigTup = []
    ratItemPredTup = []
    for item, rat in itemsRat.iteritems():
      ratItemOrigTup.append((rat, item))
      ratItemPredTup.append((np.dot(uFac[u], iFac[item]), item))
    ratItemPredTup.sort()
    ratItemOrigTup.sort()
    
    #look at Top-N items i.e. fraction of actual Top-N items in predicted Top-N
    #items
    origTopN = set(map(lambda x: x[1], ratItemOrigTup[:N]))
    predTopN = set(map(lambda x: x[1], ratItemPredTup[:N]))
    hitItems = origTopN & predTopN
    userHitRate = float(len(hitItems))/float(N)
    
    avgHitRate += userHitRate

  return avgHitRate/nUsers
    

def getItemsRat(csrMatName):
  u = 0;
  itemD = {}
  with open(csrMatName, 'r') as f:
    for line in f:
      itemD[u] = {} 
      cols = line.strip().split()
      for i in range(0, len(cols), 2):
        item = int(cols[i])
        itemRat = float(cols[i+1])
        itemD[u][item] = itemRat 
      u += 1
  return itemD



def main():
  testMatName = sys.argv[1]
  uFacName = sys.argv[2]
  iFacName = sys.argv[3]

  uFac = np.loadtxt(uFacName)
  iFac = np.loadtxt(iFacName)

  testItemsRat = getItemsRat(testMatName)
  print 'Hit rate: ', getHR(uFac, iFac, testItemsRat)


if __name__ == '__main__':
  main()
