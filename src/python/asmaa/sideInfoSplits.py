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

  
def getDim(matName):
  u = 0
  maxItem = 0
  with open(matName, 'r') as f:
    for line in f:
      u += 1
      cols = line.strip().split()
      for i in range(0, len(cols), 2):
        if int(cols[i]) > maxItem:
          maxItem = int(cols[i])
  return (u, maxItem)


def getTestValItems(nItems, testPc = 0.2, valPc=0.2):
  nTestItems = testPc*nItems
  nValItems  = valPc*nItems
  
  testItems = set([])
  valItems = set([])

  while len(testItems) < nTestItems:
    item = random.randint(0, nItems-1)
    testItems.add(item)

  while len(valItems) < nTestItems:
    item = random.randint(0, nItems-1)
    if item not in testItems:
      valItems.add(item)
  
  return (testItems, valItems)


def writeItems(items, opName):
  with open(opName, 'w') as g:
    for item in items:
      g.write(str(item) + '\n')


def createUFeatSet(ratMatName, nUsers, featDim, itemFeats, testItems, 
    valItems, opPrefix):
  
  u = 0
  
  nItems = len(itemFeats)
  nTrItems = nItems - len(testItems) - len(valItems)
  
  opPrefix = opPrefix + '_' + str(nUsers) + '_' + str(featDim)
  setOpName  = opPrefix + '_' + str(nTrItems) + '_tr.featSet'
  valOpName  = opPrefix + '_' + str(len(valItems)) + '_val.csr'
  testOpName = opPrefix + '_' + str(len(testItems)) + '_test.csr'
  maxFeat = 0

  with open(ratMatName, 'r') as f, open(setOpName, 'w') as g, \
      open(valOpName, 'w') as h, open(testOpName, 'w') as t:
    for line in f:
      cols = line.strip().split()
      uItems = []
      uRats = []
      uFeats = set([])
      for i in range(0, len(cols), 2):
        
        item = int(cols[i]) - 1
        rat = float(cols[i+1])
        
        if item not in testItems and item not in valItems:
          uItems.append(item)
          uRats.append(rat)
          uFeats = uFeats | set(itemFeats[item])
        
        if item in testItems:
          t.write(str(item) + ' ' + str(rat) + ' ')

        if item in valItems:
          h.write(str(item) + ' ' + str(rat) + ' ')
      
      uFeats = list(uFeats)
      uFeats.sort()
      if len(uFeats) and maxFeat < max(uFeats):
        maxFeat = max(uFeats)

      #write user nItems nFeats featIds
      if len(uItems) == 0:
        print 'Err: no items for ', u

      g.write(str(u) + ' ' +  str(len(uItems)) + ' ' + str(len(uFeats)) + ' ' +
          ' '.join(map(str, uFeats)) + '\n')
      for i in range(len(uItems)):
        g.write(str(uRats[i]) + ' ' + str(len(itemFeats[uItems[i]])) + ' ' 
            + ' '.join(map(str, itemFeats[uItems[i]])) + '\n')

      t.write('\n')
      h.write('\n')
      u += 1
  print 'maxFeat: ', maxFeat


def createUFeatCSR(ratMatName, nUsers, featDim, itemFeats, testItems, 
    valItems, opPrefix):
  u = 0
  nItems = len(itemFeats)
  nTrItems = nItems - len(testItems) - len(valItems)
  opPrefix = opPrefix + '_' + str(nUsers) + '_' + str(featDim)
  setOpName  = opPrefix + '_' + str(nTrItems) + '_tr.csr'
  valOpName  = opPrefix + '_' + str(len(valItems)) + '_val.csr'
  testOpName = opPrefix + '_' + str(len(testItems)) + '_test.csr'
  with open(ratMatName, 'r') as f, open(setOpName, 'w') as g, \
      open(valOpName, 'w') as h, open(testOpName, 'w') as t:
    for line in f:
      cols = line.strip().split()
      uItems = []
      uRats = []
      uFeats = set([])
      for i in range(0, len(cols), 2):
        item = int(cols[i]) - 1
        rat = float(cols[i+1])
        if item not in testItems and item not in valItems:
          g.write(str(item) + ' ' + str(rat) + ' ')
        if item in testItems:
          t.write(str(item) + ' ' + str(rat) + ' ')
        if item in valItems:
          h.write(str(item) + ' ' + str(rat) + ' ')
      g.write('\n') 
      t.write('\n')
      h.write('\n')
      u += 1


def main():
  ratMatName  = sys.argv[1]
  featMatName = sys.argv[2]
  opPrefix    = sys.argv[3]

  (nUsers, nItems) = getDim(ratMatName) 
  (nIFeat, featDim) = getDim(featMatName)
  
  print 'nUsers: ', nUsers 
  print 'nItems: ', nItems
  print 'nIFeat: ', nIFeat
  print 'featDim: ', featDim

  if (nItems != nIFeat):
    print 'Err: no. of items are not equal'

  #read movie features
  itemFeats = featMat(featMatName)  
  
  (testItems, valItems) = getTestValItems(nItems)
  createUFeatCSR(ratMatName, nUsers, featDim, itemFeats, testItems, 
      valItems, opPrefix)
  
  writeItems(testItems, opPrefix + '_testItems.txt')
  writeItems(valItems, opPrefix + '_valItems.txt')

if __name__ == '__main__':
  main()


