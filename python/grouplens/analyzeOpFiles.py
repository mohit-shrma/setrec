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
  Ns = [1, 5, 10, 25, 50, 100]

  ds = []
  
  precDics    = []
  oneCallDics = []
  for n in Ns:
    precDics.append({})
    #ds.append(precDics[-1])
    oneCallDics.append({})
    #ds.append(oneCallDics[-1])
  
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
  #ds.append(topValSetBPRDic)
  topTestSetBPRDic       = {}
  #ds.append(topTestSetBPRDic)

  topValItemBPRDic      = {}
  #ds.append(topValItemBPRDic)
  topTestItemBPRDic      = {}
  #ds.append(topTestItemBPRDic)

  topTestItemPairDic = {}
  #ds.append(topTestItemPairDic)

  keys = set([])
  fileCount = 0
  with open(ipFName, 'r') as f:
    for line in f:
      fName = line.strip()
      fileCount += 1
      if not os.path.isfile(fName):
        continue
      #if fileCount % 500 == 0:
      #  print 'Processed files ...', fileCount 
      with open(fName, 'r') as h:
        bName = os.path.basename(fName)
        bk = bName.strip('.txt').split('_')
        bk = ' '.join(bk)
        keys.add(bk)
        for fLine in h:
          
          """ 
          if fLine.startswith('Precision:'):
            cols = fLine.strip().split()
            for i in range(len(Ns)):
              updateDic(precDics[i], bk, cols[i+1])
          
          if fLine.startswith('OneCall:'):
            cols = fLine.strip().split()
            for i in range(len(Ns)):
              updateDic(oneCallDics[i], bk, cols[i+1])
          """

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
          
          """ 
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
          """

  for d in ds:
    averageDic(d)

  notFoundK = set([])

  topTestItemPairDicNF = 0
  
  for d in ds:
    for k in keys:
      if k not in d:
        print 'Not found: ', k 
        notFoundK.add(k)
  for k in keys:
    if k not in topTestItemPairDic:
      topTestItemPairDicNF += 1
  
  if len(notFoundK) > 0:
    print 'not found Count: ', len(notFoundK)
    print 'topTestK: ', topTestItemPairDicNF
    for nf in notFoundK:
      print nf
    return 

  for k in keys:
    if k in notFoundK:
      continue
    tempL = [k]
    """
    for precD in precDics:
      tempL.append(precD[k][0])
    for oneCallD in oneCallDics:
      tempL.append(oneCallD[k][0])
    """
    tempL += [trainRMSEDic[k][0], valRMSEDic[k][0], testRMSEDic[k][0],
        testAllRMSEDic[k][0]]
    tempL += [trainSRMSEDic[k][0], valSRMSEDic[k][0], testSRMSEDic[k][0]]
    #tempL += [topValSetBPRDic[k][0], topTestSetBPRDic[k][0]]
    #tempL += [topValItemBPRDic[k][0], topTestItemBPRDic[k][0]]
    #tempL += [topTestItemPairDic[k][0]]
    #tempL += [topTestItemPairDic[k][1]]
    tempL += [testRMSEDic[k][1]]
    print ' '.join(map(str, tempL))
  


def main():
  ipFName = sys.argv[1]
  parseFilesForRes(ipFName)


if __name__ == '__main__':
  main()


