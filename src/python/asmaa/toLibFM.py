import sys
import numpy as np

#takes 1-indexed feature matrix
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


#takes 1-indexed feature matrix
def featMatWVal(featMatName):
  iFeats = {}
  i = 0 #0 indexed
  with open(featMatName, 'r') as f:
    for line in f:
      feat = []
      cols = line.strip().split()
      for j in range(0, len(cols), 2):
        feat.append((int(cols[j])-1, float(cols[j+1]))) #0 indexed
        feat.sort()
      iFeats[i] = feat
      i += 1
  return iFeats


#asuming ui csr is 0-indexed
def convToLibFM(uiCSRName, nUsers, iFeats, opPrefix):
  fmOpName = opPrefix + '.libfm'
  with open(uiCSRName, 'r') as f, open(fmOpName, "w") as g:
    u = 0
    for line in f:
      cols = line.strip().split()
      for i in range(0, len(cols), 2):
        item = int(cols[i])
        rat  = cols[i+1]
        g.write(rat + ' ' + str(u)+':1 ')
        for (fInd, fVal) in iFeats[item]:
          g.write(str(nUsers+fInd) + ':' + str(fVal) + ' ')
        g.write('\n')
      u += 1


def main():
  uiCSRName   = sys.argv[1]
  featMatName = sys.argv[2]
  nUsers      = int(sys.argv[3])
  opPrefix    = sys.argv[4]

  iFeats = featMatWVal(featMatName)
  convToLibFM(uiCSRName, nUsers, iFeats, opPrefix)


if __name__ == '__main__':
  main()


