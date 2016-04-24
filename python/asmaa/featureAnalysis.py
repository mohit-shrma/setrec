import sys
import numpy as np
import random

def featMat(featMatName):
  iFeats = {}
  i = 0 #0 indexed
  with open(featMatName, 'r') as f:
    for line in f:
      feat = []
      cols = map(int, line.strip().split())
      for j in range(0, len(cols), 2):
        feat.append(int(cols[j])-1) #0 indexed
        feat.sort()
      iFeats[i] = feat
      i += 1
  return iFeats


def getTestItems(testItemsFName):
  testItems = [] 
  with open(testItemsFName, 'r') as f:
    for line in f:
      testItems.append(int(line.strip()))
  return testItems


def rating(u, item, itemFeats, uFac, fFac):
  est_rat = 0
  itemFeat = itemFeats[item]
  for feat in itemFeat:
    est_rat += np.dot(uFac[u], fFac[feat])
  est_rat = est_rat/len(itemFeat)
  return est_rat



def getTopFeats(uFac, fFac, ignoreUs, opName):
  nUsers = uFac.shape[0]
  nFeatures = fFac.shape[0]
  uTop50 = {}
  with open(opName, "w") as g:
    for u in range(nUsers):
      uFVals = []
      for fInd in range(nFeatures):
        uFVals.append((np.dot(uFac[u], fFac[fInd]), fInd))
      uFVals.sort(reverse=True)
      uTop50[u] = uFVals[:10]
      g.write(str(u) + ' ')
      for (fVal, fInd) in uFVals[:10]:
        g.write(str(fInd) + ' ' + str(fVal) + ' ')
      g.write('\n')
  return uTop50


def getPreds(uFac, fFac, itemFeats, ignoreUs, testItems, opName):
  nUsers = uFac.shape[0]
  predUs = set([])
  with open(opName, "w") as g:
    while len(predUs) < 10:
      u = random.randint(0, nUsers-1)
      if u in predUs or u in ignoreUs:
        continue
      predUs.add(u)
      ratItems = []
      for item in testItems:
        rat = 0.0
        for fInd in itemFeats[item]:
          rat += np.dot(uFac[u], fFac[fInd])
        rat = rat/len(itemFeats[item])
        ratItems.append((rat, item))
      ratItems.sort(reverse = True)
      for (rat, item) in ratItems:
        g.write(str(u) + ',' + str(item) + ',' + str(rat) +  ',' +
            str(len(itemFeats[item])) + '\n')


def main():
  uFacName       = sys.argv[1]
  fFacName       = sys.argv[2]
  featMatName    = sys.argv[3] 
  testItemsFName = sys.argv[4]
  ignoreUName    = sys.argv[5]

  uFac      = np.loadtxt(uFacName)
  fFac      = np.loadtxt(fFacName)
  nUsers    = uFac.shape[0]
  nFeatures = fFac.shape[0]

  print 'nUsers: ', nUsers
  testItems = getTestItems(testItemsFName)
  itemFeats = featMat(featMatName)
  ignoreUs = getTestItems(ignoreUName)
  
  #getTopFeats(uFac, fFac, ignoreUs, 'uTopFeat.txt')
  getPreds(uFac, fFac, itemFeats, ignoreUs, testItems, 'uTestPreds.txt')

if __name__ == '__main__':
  main()





