import sys
import numpy as np


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
  setsFName = sys.argv[1]
  uLatFacFName = sys.argv[2]
  iLatFacFName = sys.argv[3]
  
  iFac = np.loadtxt(iLatFacFName)
  uFac = np.loadtxt(uLatFacFName)
  
  ratingsOnSet(setsFName, uFac, iFac)
  #rmse = ratingOnItems(itemsFName, uFac, iFac)
  #print 'Test rmse: ', rmse


if __name__ == '__main__':
  main()


