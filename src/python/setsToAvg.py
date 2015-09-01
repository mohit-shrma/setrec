import sys
import numpy as np


def toAvgRat(fName, uFac, iFac, opName, nUsers = 1000):
  with open(fName, 'r') as f, open(opName, 'w') as g:
    u = 0
    while u < nUsers:
      uHead = f.readline()
      g.write(uHead)
      cols = uHead.strip().split()
      nSets = int(cols[1])
      for i in range(nSets):
        setStr = f.readline()
        cols = setStr.strip().split()
        nItems = int(cols[1])
        items = map(int, cols[2:])
        sm = 0.0
        for item in items:
          sm += np.dot(uFac[u], iFac[item])
        setRat = sm/nItems
        g.write("%.3f" % setRat + ' ' + str(nItems) + ' ' + ' '.join(cols[2:])
            + '\n')
      u += 1


def main():
  fName = sys.argv[1]
  uFacName = sys.argv[2]
  iFacName = sys.argv[3]
  opName = sys.argv[4]


  uFac = np.loadtxt(uFacName)
  iFac = np.loadtxt(iFacName)

  toAvgRat(fName, uFac, iFac, opName)

if __name__ == '__main__':
  main()


