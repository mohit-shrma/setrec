import sys
import random
import numpy as np



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


def createUFeatSet(ratMatName, itemFeats, testItems, valItems, opPrefix):
  u = 0
  setOpName  = opPrefix + '_tr.featSet'
  valOpName  = opPrefix + '_val.featSet'
  testOpName = opPrefix + '_test.featSet'
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
          t.write(str(u) + ' ' + str(item) + ' ' + str(rat) + ' ' 
              + ' '.join(map(str, itemFeats[item])) + '\n')

        if item in valItems:
          h.write(str(u) + ' ' + str(item) + ' ' + str(rat) + ' ' 
              + ' '.join(map(str, itemFeats[item])) + '\n')
      
      uFeats = list(uFeats)
      uFeats.sort()
      #write user nItems nFeats featIds
      g.write(str(u) + ' ' +  str(len(uItems)) + ' ' + str(len(uFeats)) + ' ' +
          ' '.join(map(str, uFeats)) + '\n')
      for i in range(len(uItems)):
        g.write(str(uRats[i]) + ' ' + str(len(itemFeats[uItems[i]])) + ' ' 
            + ' '.join(map(str, itemFeats[uItems[i]])) + '\n')
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
  createUFeatSet(ratMatName, itemFeats, testItems, valItems, 
      opPrefix + '_' + str(nUsers) + '_' + str(nItems) + '_' + str(featDim))


if __name__ == '__main__':
  main()


