import sys
import numpy as np


def genScaledFacs(nUsers, nItems, dim, scale = 80000):
  A = np.random.rand(nUsers, dim)
  B = np.random.rand(nItems, dim)
  [ua, sa, va] = np.linalg.svd(A, full_matrices=0)
  [ub, sb, vb] = np.linalg.svd(B, full_matrices=0)
  S = np.identity(dim)*np.sqrt(scale)
  uFac = np.dot(ua, S)
  iFac = np.dot(ub, S)
  print 'uFac Norm: ', np.linalg.norm(uFac)
  print 'iFac Norm: ', np.linalg.norm(iFac)
  X = np.dot(uFac[np.random.randint(nUsers, size=500), :],
      iFac[np.random.randint(nItems, size=500), :].T)
  print 'avg: ', np.average(X)
  print 'min: ', np.min(X)
  print 'max: ', np.max(X)
  uFacName = 'uFac_' + str(nUsers) + '_' + str(dim) + '.txt'
  iFacName = 'iFac_' + str(nItems) + '_' + str(dim) + '.txt'
  np.savetxt(uFacName, uFac)
  np.savetxt(iFacName, iFac)
  return (uFac, iFac)


""" 
user's distribution to select from 9 choices
0 a[0] /1
1 a[0...1] /2
2 a[0...2] /3
3 a[0...3] /4
4 a[0...4] /5
5 a[1...4] /4
6 a[2...4] /3
7 a[3...4] /2
8 a[4] /1
"""
def generateUserPickynessDist(nUsers, opFName, nChoices=9):
  pickyDist = {}
  with open(opFName, 'w') as g:
    for u in range(nUsers):
      a = np.random.rand(nChoices)
      pDist = a/sum(a)
      pickyDist[u] = pDist
      g.write(' '.join(map(str, pDist)) + '\n')
  return pickyDist


def selectChoiceFromDist(dist):
  choice = np.random.choice(np.arange(len(dist)), p = dist)
  return choice


def setRatingByChoice(sortedRatings, choice):
  if choice == 0:
    return sortedRatings[0]
  elif choice == 1:
    return sum(sortedRatings[:2])/2
  elif choice == 2:
    return sum(sortedRatings[:3])/3
  elif choice == 3:
    return sum(sortedRatings[:4])/4
  elif choice == 4:
    return sum(sortedRatings[:5])/5
  elif choice == 5:
    return sum(sortedRatings[1:5])/4
  elif choice == 6:
    return sum(sortedRatings[2:5])/3
  elif choice == 7:
    return sum(sortedRatings[3:5])/2
  elif choice == 8:
    return sortedRatings[4]
  else:
    print "ERROR: invalid choice found"


def weightedSetRating(sortedRatings, pickyDist):
  pass


def setRating(user, items, uFacs, iFacs, pickyDist):
  ratings = []
  for item in items:
    ratings.append(np.dot(uFacs[user], iFacs[item]))
  ratings.sort()
  uChoice = selectChoiceFromDist(pickyDist[user])
  return setRatingByChoice(ratings, uChoice) 


#read original sets and write synthetic sets according to dist
def writeSynSets(ipFileName, opFileName, uFacs, iFacs, pickyDist):
  nUsers = uFacs.shape[0]
  nItems = iFacs.shape[1]
  with open(ipFileName, 'r') as f, open(opFileName, 'w') as g:
    isEnd = False
    while(not isEnd):
      line = f.readline()
      if len(line) == 0:
        isEnd = True
        break
      cols = line.strip().split()
      cols = map(int, cols)
      user = cols[0]
      nSets = cols[1]
      nItems = len(cols[2:])
      
      if user >= nUsers:
        print 'Invalid user: ', user, ' found'

      #write user line as it is
      g.write(line)
      
      for k in range(nSets):
        setLine = f.readline()
        if len(setLine) == 0:
          isEnd = True
          break
        setLineCols = setLine.strip().split()
        setItems = map(int, setLineCols[2:])
        if len(setItems) != 5:
          print 'Invalid set found'
        rating = setRating(user, setItems, uFacs, iFacs, pickyDist)
        g.write(str(rating) + ' 5 ' + ' '.join(setLineCols[2:]) + '\n')


def writeSynCSR(ipFileName, opFileName, nUsers, nItems, uFacs, iFacs):
  u = 0
  with open(ipFileName, 'r') as f, open(opFileName, 'w') as g:
    for line in f:
      cols = line.strip().split()
      if len(cols) > 0:
        for i in range(0, len(cols), 2):
          item = int(cols[i])
          g.write(str(item) + ' ' + str(np.dot(uFacs[u], iFacs[item])) + ' ')
      g.write('\n')
      u += 1


def main():

  nUsers = int(sys.argv[1])
  nItems = int(sys.argv[2])
  facDim = int(sys.argv[3])
  scale  = int(sys.argv[4])
  randSeed = 1
  np.random.seed(randSeed)

  (uFacs, iFacs) = genScaledFacs(nUsers, nItems, facDim, scale)
  pickyDist = generateUserPickynessDist(nUsers, "userPickyness.txt")
  
  writeSynSets("ml_set.lfs", "ml_set.syn.lfs", uFacs, iFacs, pickyDist)
  writeSynSets("ml_set.train.lfs", "ml_set.train.syn.lfs", uFacs, iFacs, pickyDist)
  writeSynSets("ml_set.test.lfs", "ml_set.test.syn.lfs", uFacs, iFacs, pickyDist)
  writeSynSets("ml_set.val.lfs", "ml_set.val.syn.lfs", uFacs, iFacs, pickyDist)

  writeSynCSR("ml_ratings.csr", "ml_ratings.syn.csr", nUsers, nItems, uFacs,
      iFacs)
  writeSynCSR("train.csr", "train.syn.csr", nUsers, nItems, uFacs,
      iFacs)
  writeSynCSR("test.csr", "test.syn.csr", nUsers, nItems, uFacs,
      iFacs)
  writeSynCSR("val.csr", "val.syn.csr", nUsers, nItems, uFacs,
      iFacs)


if __name__ == '__main__':
  main()


