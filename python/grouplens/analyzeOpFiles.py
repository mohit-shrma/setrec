import sys
import os

def updateDic(dic, k, line):
  cols = line.strip().split()
  val = float(cols[-1])
  if k not in dic:
    dic[k] = [0.0, 0.0]
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
  
  ordItemPairDic     = {}
  ordSetPairDic      = {}
  ordTestItemPairDic = {}

  valRMSEDic     = {}
  testRMSEDic    = {}
  trainRMSEDic   = {}
  
  valSRMSEDic    = {}
  testSRMSEDic   = {}
  trainSRMSEDic  = {}
 
  topSetBPRDic       = {}
  topItemBPRDic      = {}
  topTestItemPairDic = {}

  keys = set([])
  with open(ipFName, 'r') as f:
    for line in f:
      fName = line.strip()
      with open(fName, 'r') as h:
        bName = os.path.basename(fName)
        bk = bName.strip('.txt').split('_')
        bk = ' '.join(bk)
        keys.add(bk)
        for fLine in h:
          
          if fLine.startswith('Precision@5'): 
            updateDic(prec5Dic, bk, fLine)
          if fLine.startswith('Precision@10'):
            updateDic(prec10Dic, bk, fLine)
          
          if fLine.startswith('OneCall@5'):
            updateDic(oneCall5Dic, bk, fLine)
          if fLine.startswith('OneCall@10'):
            updateDic(oneCall10Dic, bk, fLine)

          if fLine.startswith('Fraction of correct ordered set'):
            updateDic(ordSetPairDic, bk, fLine)
          if fLine.startswith('Fraction of correct ordered item'):
            updateDic(ordItemPairDic, bk, fLine)
          if fLine.startswith("Ordered pairs exc"):
            updateDic(ordTestItemPairDic, bk, fLine)

          if fLine.startswith("Val RM"):
            updateDic(valRMSEDic, bk, fLine)
          if fLine.startswith("Test RM"):
            updateDic(testRMSEDic, bk, fLine)
          if fLine.startswith("Train RM"):
            updateDic(trainRMSEDic, bk, fLine)

          if fLine.startswith("Val set"):
            updateDic(valSRMSEDic, bk, fLine)
          if fLine.startswith("Test set"):
            updateDic(testSRMSEDic, bk, fLine)
          if fLine.startswith('Train set') or 'Train sets RMSE:' in fLine:
            updateDic(trainSRMSEDic, bk, fLine)

          if fLine.startswith("Fraction top correct ordered set"):
            updateDic(topSetBPRDic, bk, fLine)
          if fLine.startswith("Fraction top item"):
            updateDic(topItemBPRDic, bk, fLine)
          if fLine.startswith("Ordered top pairs"):
            updateDic(topTestItemPairDic, bk, fLine)

  ds = []
 
  averageDic(prec5Dic)  
  ds.append(prec5Dic)

  averageDic(prec10Dic)
  ds.append(prec10Dic)

  averageDic(oneCall5Dic)
  ds.append(oneCall5Dic)

  averageDic(oneCall10Dic)
  ds.append(oneCall10Dic)

  averageDic(ordItemPairDic)
  ds.append(ordItemPairDic)

  averageDic(ordSetPairDic)
  ds.append(ordSetPairDic)

  averageDic(ordTestItemPairDic)
  ds.append(ordTestItemPairDic)

  averageDic(valRMSEDic)
  ds.append(valRMSEDic)

  averageDic(testRMSEDic)
  ds.append(testRMSEDic)

  averageDic(trainRMSEDic)
  ds.append(trainRMSEDic)

  averageDic(valSRMSEDic)
  ds.append(valSRMSEDic)

  averageDic(testSRMSEDic)
  ds.append(testSRMSEDic)

  averageDic(trainSRMSEDic)
  ds.append(trainSRMSEDic)

  averageDic(topSetBPRDic)
  ds.append(topSetBPRDic)

  averageDic(topItemBPRDic)
  ds.append(topItemBPRDic)

  averageDic(topTestItemPairDic)
  ds.append(topTestItemPairDic)

  for d in ds:
    for k in keys:
      if k not in d:
        print 'Not found: ', k 
        return 


  for k in keys:
    print k + ' ' + str(prec5Dic[k][0]) + ' ' + str(prec10Dic[k][0]) + \
        ' ' + str(oneCall5Dic[k][0]) + ' ' + str(oneCall10Dic[k][0]) + \
        ' ' + str(trainRMSEDic[k][0]) + ' ' + str(valRMSEDic[k][0]) + \
        ' ' + str(testRMSEDic[k][0]) + \
        ' ' + str(trainSRMSEDic[k][0]) + ' ' + str(valSRMSEDic[k][0]) + \
        ' ' + str(testSRMSEDic[k][0]) + \
        ' ' + str(ordSetPairDic[k][0]) + ' ' + str(ordItemPairDic[k][0]) + \
        ' ' + str(ordTestItemPairDic[k][0]) + \
        ' ' + str(topSetBPRDic[k][0]) + ' ' + str(topItemBPRDic[k][0]) + \
        ' ' + str(topTestItemPairDic[k][0]) + \
        ' ' + str(prec5Dic[k][1]) 
  


def main():
  ipFName = sys.argv[1]
  parseFilesForRes(ipFName)


if __name__ == '__main__':
  main()


