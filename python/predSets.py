import sys
import numpy as np


def ratingsOnTrain(trainFName, uFac, iFac, nUsers = 1000):
  
  with open (trainFName, 'r') as f:
    
    u = 0
    rmse = 0.0 
    nTotalSets = 0
    
    #outItems = set([440, 601, 696, 770, 303, 870, 926])
    outItems = set([])

    while u < nUsers:
      
      uHead = f.readline()  
      cols = uHead.strip().split()
      user = int(cols[0])
      nSets = int(cols[1])
      setItems = map(int, cols[2:])
      
      for i in range(nSets):
        setStr = f .readline()
        cols = setStr.strip().split()
        setRat = float(cols[0])
        nItems = int(cols[1])
        items = map(int, cols[2:])
        
        if len(set(items) & outItems) > 0:
          continue

        itemRats = []
        for item in items:
          itemRats.append((np.dot(uFac[user], iFac[item]), item))
        itemRats.sort(reverse = True)

        majSz = len(items)/2
        if len(items) % 2 != 0:
          majSz = len(items)/2 + 1
        
        sm = 0.0
        for k in range(majSz):
          sm += itemRats[k][0]
        estSetRat = sm/majSz
        
        rmse += (estSetRat - setRat)*(estSetRat - setRat)
        nTotalSets += 1

      u += 1
    
    rmse = np.sqrt(rmse/nTotalSets)
    
    print 'rmse: ', rmse


def ratingsOnSet(setsFName, uFac, iFac):
  
  with open(setsFName, 'r') as f, open('setRatings.txt', 'w') as g:
    u = -1
    for line in f:
      u += 1
      cols = line.strip().split()
      items = map(int, cols)
      itemRats = []
      
      for item in items:
        rat = np.dot(uFac[u], iFac[item]) 
        itemRats.append((rat,item))
      
      majSz = len(itemRats)/2
      if len(itemRats) % 2 != 0:
        majSz = len(itemRats)/2 + 1
      
      itemRats.sort(reverse=True)
     
      sm = 0
      for i in range(majSz):
        sm += itemRats[i][0]

      setRat = sm/majSz
      
      g.write(str(setRat) + '\n')


def ratingOnItems(itemsFName, uFac, iFac):
  with open(itemsFName, 'r') as f, open('itemRats.txt', 'w') as g:
    u = -1
    sm = 0
    for line in f:
      u += 1
      cols = line.strip().split()
      item = int(cols[0])
      rat2 = float(cols[1])
      rat1 = np.dot(uFac[u], iFac[item])
      sm += (rat2 - rat1)*(rat2 - rat1)
      g.write(str(rat1) + '\n')
    return np.sqrt(sm/u)


def main():
  fName = sys.argv[1]
  uLatFacFName = sys.argv[2]
  iLatFacFName = sys.argv[3]
  
  iFac = np.loadtxt(iLatFacFName)
  uFac = np.loadtxt(uLatFacFName)
  
  ratingsOnTrain(fName, uFac, iFac)

  #ratingsOnSet(setsFName, uFac, iFac)
  #rmse = ratingOnItems(itemsFName, uFac, iFac)
  #print 'Test rmse: ', rmse


if __name__ == '__main__':
  main()


