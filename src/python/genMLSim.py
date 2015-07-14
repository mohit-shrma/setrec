import numpy as np
import sys

""" 
this will compute similarities given the map features and orig movie details
python ~/dev/bitbucket/setrec/src/python/genMLSim.py  rat_set_5_4.0_50_i_map
movieGenreVec.csv movies.csv testSim 20
"""


def readMap(ipName):
  iMap = {}
  revIMap = {}
  with open(ipName, 'r') as f:
    for line in f:
      cols = line.strip().split()
      iMap[int(cols[0])] = int(cols[1])
      revIMap[int(cols[1])] = int(cols[0])
  return (iMap, revIMap)


def getFeatures(ipName, nFeatures = 20):
  nItems = 0
  with open(ipName, 'r') as f:
    for line in f:
      nItems += 1

  print "nItems: ", nItems

  iFeatureArr = np.zeros((nItems, nFeatures))

  with open(ipName, 'r') as f:
    i = 0
    for line in f:
      cols = line.strip().split()
      cols = map(int, cols)
      for ind in cols:
        iFeatureArr[i][ind] = 1.0
      i += 1 
  return iFeatureArr


def readMovieIds(ipName):
  movIds = []
  with open(ipName, 'r') as f:
    f.readline()
    for line in f:
      cols = line.strip().split(',')
      movIds.append(int(cols[0]))
  return movIds


def writeSim(origItemOrdKeys, iFeatureArr, movDic, iMap, opName):
  nItems = len(origItemOrdKeys)
  print "Items to be considered for computing sim: ", nItems
  with open(opName, 'w') as g:
    for i in range(len(origItemOrdKeys)):
      for j in range(i+1, len(origItemOrdKeys)):
        item1 = origItemOrdKeys[i]
        item2 = origItemOrdKeys[j]
        ind1 = movDic[item1]
        ind2 = movDic[item2]
        pairSim = 0
        norm1 = np.linalg.norm(iFeatureArr[ind1]) 
        norm2 = np.linalg.norm(iFeatureArr[ind2]) 
        if norm1 > 0 and norm2 > 0:
          pairSim = np.dot(iFeatureArr[ind1], iFeatureArr[ind2])
          pairSim = pairSim/(norm1*norm2)
        g.write(str(iMap[item1]) + ' ' +  str(iMap[item2]) + ' ' + str(pairSim) + '\n')


def getOrigOrderKeys(revIMap):
  newOrigTups = []
  for newItem, origItem in revIMap.iteritems():
    newOrigTups.append((newItem, origItem))
  newOrigTups.sort()
  origOrderKey = []
  for (n, o) in newOrigTups:
    origOrderKey.append(o)
  return origOrderKey
  

def main():
  
  mapName   = sys.argv[1]
  featName  = sys.argv[2]
  movFName  = sys.argv[3]
  opName    = sys.argv[4]
  nFeatures = int(sys.argv[5])
  

  (iMap, revIMap) = readMap(mapName)
  iFeatureArr     = getFeatures(featName, nFeatures)
  origOrderKey    = getOrigOrderKeys(revIMap)
  movIds          = readMovieIds(movFName)

  movDic = dict(zip(movIds, range(len(movIds))))

  writeSim(origOrderKey, iFeatureArr, movDic, iMap, opName)



  
if __name__ == '__main__':
  main()
  



