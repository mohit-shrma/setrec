import sys


def averageParams(splitFNames):
  avgDic = {}
  for fName in splitFNames:
    with open (fName, 'r') as f:
      for line in f:
        cols = line.strip().split()
        k = '_'.join(cols[:-1])
        val = float(cols[-1])
        if k not in avgDic:
          avgDic[k] = [0.0, 0.0]
        avgDic[k][0] += 1
        avgDic[k][1] += val
  for k, countVal in avgDic.iteritems():
    avgDic[k] = [countVal[0], countVal[1]/countVal[0]]
  return avgDic


def writeAvgDic(avgDic, opFName):
  with open(opFName, 'w') as g:
    for k, v in avgDic.iteritems():
      strk = k.replace('_', '\t')
      g.write(strk + '\t' + str(v[1]) + '\t' + str(v[0]) + '\n')
      
      

def main():
  splitFNames = sys.argv[1:-1]
  opFName = sys.argv[-1]
  
  print 'splits: ', splitFNames
  print 'output: ', opFName

  avgDic = averageParams(splitFNames)
  writeAvgDic(avgDic, opFName)



if __name__ == '__main__':
  main()


