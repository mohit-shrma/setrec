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
  ds = []
  prec5Dic       = {}
  ds.append(prec5Dic)
  prec10Dic      = {}
  ds.append(prec10Dic)
  
  oneCall5Dic    = {}
  ds.append(oneCall5Dic)
  oneCall10Dic   = {}
  ds.append(oneCall10Dic)
  
  valRMSEDic     = {}
  ds.append(valRMSEDic)
  testRMSEDic    = {}
  ds.append(testRMSEDic)
  trainRMSEDic   = {}
  ds.append(trainRMSEDic)
  testAllRMSEDic = {}
  ds.append(testAllRMSEDic)

  valSRMSEDic    = {}
  ds.append(valSRMSEDic)
  testSRMSEDic   = {}
  ds.append(testSRMSEDic)
  trainSRMSEDic  = {}
  ds.append(trainSRMSEDic)
 
  topValSetBPRDic       = {}
  ds.append(topValSetBPRDic)
  topTestSetBPRDic       = {}
  ds.append(topTestSetBPRDic)

  topValItemBPRDic      = {}
  ds.append(topValItemBPRDic)
  topTestItemBPRDic      = {}
  ds.append(topTestItemBPRDic)

  topTestItemPairDic = {}
  ds.append(topTestItemPairDic)

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
          
          if fLine.startswith('Precision@5:'): 
            updateDic(prec5Dic, bk, fLine)
          if fLine.startswith('Precision@10:'):
            updateDic(prec10Dic, bk, fLine)
          
          if fLine.startswith('OneCall@5:'):
            updateDic(oneCall5Dic, bk, fLine)
          if fLine.startswith('OneCall@10:'):
            updateDic(oneCall10Dic, bk, fLine)

          if fLine.startswith("Val RM"):
            updateDic(valRMSEDic, bk, fLine)
          if fLine.startswith("Test RM"):
            updateDic(testRMSEDic, bk, fLine)
          if fLine.startswith("Train RM"):
            updateDic(trainRMSEDic, bk, fLine)
          if fLine.startswith('Test All Mat RM'):
            updateDic(testAllRMSEDic, bk, fLine)

          if fLine.startswith("Val set"):
            updateDic(valSRMSEDic, bk, fLine)
          if fLine.startswith("Test set"):
            updateDic(testSRMSEDic, bk, fLine)
          if fLine.startswith('Train set') or 'Train sets RMSE:' in fLine:
            updateDic(trainSRMSEDic, bk, fLine)
          
          if fLine.startswith("Fraction top val correct ordered sets"):
            updateDic(topValSetBPRDic, bk, fLine)
          if fLine.startswith("Fraction top test correct ordered sets"):
            updateDic(topTestSetBPRDic, bk, fLine)
          
          if fLine.startswith("Fraction top val item pair"):
            updateDic(topValItemBPRDic, bk, fLine)
          if fLine.startswith("Fraction top test item pairs:"):
            updateDic(topTestItemBPRDic, bk, fLine)

          if fLine.startswith("Ordered top pairs"):
            updateDic(topTestItemPairDic, bk, fLine)

  for d in ds:
    averageDic(d)

  notFoundK = set([])

  for d in ds:
    for k in keys:
      if k not in d:
        print 'Not found: ', k 
        notFoundK.add(k)
  
  if len(notFoundK) > 0:
    for nf in notFoundK:
      print nf
    return 

  for k in keys:
    if k in notFoundK:
      continue
    print k + ' ' + str(prec5Dic[k][0]) + ' ' + str(prec10Dic[k][0]) + \
        ' ' + str(oneCall5Dic[k][0]) + ' ' + str(oneCall10Dic[k][0]) + \
        ' ' + str(trainRMSEDic[k][0]) + ' ' + str(valRMSEDic[k][0]) + \
        ' ' + str(testRMSEDic[k][0]) +  ' ' +  str(testAllRMSEDic[k][0]) + \
        ' ' + str(trainSRMSEDic[k][0]) + ' ' + str(valSRMSEDic[k][0]) + \
        ' ' + str(testSRMSEDic[k][0]) + \
        ' ' + str(topValSetBPRDic[k][0]) + ' ' + str(topTestSetBPRDic[k][0]) + \
        ' ' + str(topValItemBPRDic[k][0]) + ' ' + str(topTestItemBPRDic[k][0]) + \
        ' ' + str(topTestItemPairDic[k][0]) + \
        ' ' + str(prec5Dic[k][1]) 
  


def main():
  ipFName = sys.argv[1]
  parseFilesForRes(ipFName)


if __name__ == '__main__':
  main()


