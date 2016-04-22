import sys
import random
import numpy as np


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


def evalHR(testCSRName, itemFeats, testItems, ignoreUs, nUsers, uFac, fFac):
  
  recallU = 0
  recall = 0
  relU = 0
  
  with open(testCSRName, 'r') as f:
    u = -1
    for line in f:
      u += 1
      if u  in ignoreUs:
        continue
      cols = line.strip().split()
      testItemRats = []
      for item in testItems:
        #estimate rating for item
        rat_est = 0
        itemFeat = itemFeats[item]
        for feat in itemFeat:
          rat_est += np.dot(uFac[u], fFac[feat])
        rat_est = rat_est/len(itemFeat)
        #rat_est = 1.0/(1.0 + np.exp(-rat_est))
        testItemRats.append((rat_est, item))
      testItemRats.sort(reverse = True)
      top10TestItems = set(map(lambda x: x[1], testItemRats[:10]))
      top10Rats = map(lambda x: x[0], testItemRats[:10]) 
      actTestItems = set([])
      for i in range(0, len(cols), 2):
        item = int(cols[i]) 
        actTestItems.add(item)
      
      if len(actTestItems) == 0:
        print 'No test item for u: ', u
        continue

      relU += 1
      recallU = -1
      if len(actTestItems) < 10:
        recallU = float(len(actTestItems & top10TestItems)) / float(len(actTestItems))
      else:
        recallU = float(len(actTestItems & top10TestItems)) / 10.0
      
      print 'u: ', u, 'recall: ', recallU, testItemRats[:10]

      recall += recallU
  print 'relU: ', relU 
  return recall/relU


def main():
  uFacName      = sys.argv[1]
  fFacFName      = sys.argv[2]
  testCSRName    = sys.argv[3]
  testItemsFName = sys.argv[4]
  featMatName    = sys.argv[5]
  ignoreUName    = sys.argv[6]

  uFac = np.loadtxt(uFacName)
  fFac = np.loadtxt(fFacFName)
  nUsers = uFac.shape[0]
  
  print 'nUsers: ', nUsers

  testItems = getTestItems(testItemsFName)  
  itemFeats = featMat(featMatName)  
  ignoreUs = getTestItems(ignoreUName)
  
  hr = evalHR(testCSRName, itemFeats, testItems, ignoreUs, nUsers, 
              uFac, fFac) 
  print 'HR: ', hr 


if __name__ == '__main__':
  main()

