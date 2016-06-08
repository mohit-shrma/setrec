import sys
import os

def updateDic(dic, k, line):
  cols = line.strip().split()
  val = float(cols[-1])
  if k not in dic:
    dic[k] = (0.0, 0.0)
  dic[k][0] += val
  dic[k][1] += 1


def averageDic(dic):
  for k, v in dic.iteritems():
    dic[k][0] = dic[k][0]/dic[k][1]


def parseFilesForRes(ipFName):
  
  prec5Dic       = {}
  prec10Dic      = {}
  
  oneCall5Dic    = {}
  oneCall10Dic   = {}
  
  ordItemPairDic = {}
  ordSetPairDic  = {}
  
  valRMSEDic     = {}
  testRMSEDic    = {}
  trainRMSEDic    = {}
  
  valSRMSEDic    = {}
  testSRMSEDic   = {}
  trainSRMSEDic   = {}
  
  keys = set([])
  with open(ipFName, 'r') as f:
    for line in f:
      fName = line.strip()
      with open(fName, 'r') as h:
        bName = os.path.basename(fName)
        bk = bName.strip('.txt').split('_')
        keys.add(bk)
        for fLine in h:
          
          if fLine.startswith('Precision@5'): 
            updateDic(prec5Dic, bk, line)
          if fLine.startswith('Precision@10'):
            updateDic(prec10Dic, bk, line)
          
          if fLine.startswith('OneCall@5'):
            updateDic(oneCall5Dic, bk, line)
          if fLine.startswith('OneCall@10'):
            updateDic(oneCall10Dic, bk, line)

          if fLine.startswith('Fraction of correct ordered set'):
            updateDic(ordSetPairDic, bk, line)
          if fLine.startswith('Fraction of correct ordered item'):
            updateDic(ordItemPairDic, bk, line)
          
          if fLine.startswith("Val RM"):
            updateDic(valRMSEDic, bk, line)
          if fLine.startswith("Test RM"):
            updateDic(testRMSEDic, bk, line)
          if fLine.startswith("Train RM"):
            updateDic(trainRMSEDic, bk, line)

          if fLine.startswith("Val set"):
            updateDic(valSRMSEDic, bk, line)
          if fLine.startswith("Test set"):
            updateDic(testSRMSEDic, bk, line)
          if fLine.startswith('Train set'):
            updateDic(trainSRMSEDic, bk, line)

  averageDic(prec5Dic)  
  averageDic(prec10Dic)
  averageDic(oneCall5Dic)
  averageDic(oneCall10Dic)
  averageDic(ordItemPairDic)
  averageDic(ordSetPairDic)
  averageDic(valRMSEDic)
  averageDic(testSRMSEDic)
  averageDic(trainRMSEDic)
  averageDic(valSRMSEDic)
  averageDic(testSRMSEDic)
  averageDic(trainSRMSEDic)
  
  for k in keys:
    print '\t'.join(k) + '\t' + str(prec5Dic[k][0]) + '\t' + str(prec10Dic[k][0]) + \
        '\t' + str(oneCall5Dic[k][0]) + '\t' + str(oneCall10Dic[k][0]) + \
        '\t' + str(trainRMSEDic[k][0]) + '\t' + str(valRMSEDic[k][0]) + \
        '\t' + str(testRMSEDic[k][0]) + \
        '\t' + str(trainSRMSEDic[k][0]) + '\t' + str(valSRMSEDic[k][0]) + \
        '\t' + str(testSRMSEDic[k][0]) + \
        '\t' + str(ordSetPairDic[k][0]) + '\t' + str(ordItemPairDic[k][0]) + \
        '\t' + str(prec5Dic[k][1])


def main():
  ipFName = sys.argv[1]
  parseFilesForRes(ipFName)


if __name__ == '__main__':
  main()


