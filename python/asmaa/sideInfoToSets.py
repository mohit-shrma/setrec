import numpy as np
import sys


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


#TODO: write 0 indexed
def createUFeatSet(ratMatName, itemFeats, opName):
  u = 0
  with open(ratMatName, 'r') as f, open(opName, 'w') as g:
    for line in f:
      cols = line.strip().split()
      uItems = []
      uRats = []
      uFeats = set([])
      for i in range(0, len(cols), 2):
        item = int(cols[i]) - 1
        uItems.append(item)
        rat = float(cols[i+1])
        uRats.append(rat)
        uFeats = uFeats | set(itemFeats[item])
      uFeats = list(uFeats)
      uFeats.sort()
      #write user nItems nFeats featIds
      g.write(str(u) + ' ' +  str(len(uItems)) + ' ' + str(len(uFeats)) + ' ' +
          ' '.join(map(str, uFeats)) + '\n')
      for i in range(len(uItems)):
        g.write(str(uRats[i]) + ' ' + str(len(itemFeats[uItems[i]])) + ' ' 
            + ' '.join(map(str, itemFeats[uItems[i]])) + '\n')
      u += 1


#NOTE: matrices are 1-indexed
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

  #create user-movie sets
  createUFeatSet(ratMatName, itemFeats, opPrefix + '_' + str(nUsers) + '_' +
      str(nItems) + '_' + str(featDim) + '.featSet')


if __name__ == '__main__':
  main()


