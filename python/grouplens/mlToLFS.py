import sys
import random

UNDER_RATED_SET   = 1
OVER_RATED_SET    = 2
NEITHER_RATED_SET = 0


def avgSetRat(user, items, uiRatings):
  rat = 0.0
  for item in items:
    rat += uiRatings[user][item]
  rat = rat/len(items)
  return rat


def isSetUnderOverRat(user, items, rating, uiRatings):
  avgRat = avgSetRat(user, items, uiRatings)
  if avgRat >= rating + 0.5:
    #under rat
    return UNDER_RATED_SET
  elif avgRat <= rating - 0.5:
    #over rat
    return OVER_RATED_SET
  else:
    return NEITHER_RATED_SET


def checkRatingsExist(uSetRatings, uiRatings):
  for user, setRatings in uSetRatings.iteritems():
    for setRating in setRatings:
      uItems = setRating[0]
      for item in uItems:
        if item not in uiRatings[user]:
          print 'Rating not found: ', user, item 


def getUItems(setFName):
  users = set([])
  items = set([])
  with open(setFName, 'r') as f:
    for line in f:
      cols   = line.strip().split(',')
      user   = int(cols[0])
      uItems = map(int, cols[1].split('-'))
      for item in uItems:
        items.add(item)
      users.add(user)
  print 'nUsers: ', len(users), ' nItems: ', len(items)
  return (users, items)


def getMap(ids):
  return dict(zip(ids, range(len(ids))))


def writeMap(m, opFName):
  with open(opFName, 'w') as g:
    for k, v in m.iteritems():
      g.write(str(k) + ',' + str(v) + '\n')


def getUIRatings(ipRatName):
  uiRatings = {}
  uinnz = 0
  with open(ipRatName, 'r') as f:
    for line in f:
      cols = line.strip().split(',')
      user = int(cols[0])
      item = int(cols[1])
      rating = float(cols[2])
      if user not in uiRatings:
        uiRatings[user] = {}
      uiRatings[user][item] = rating
      uinnz += 1
  print 'nRatings: ', uinnz
  return uiRatings


def writeUIRatingsCSR(uiRatings, opFName, uMap, iMap):
  revUMap = {}
  for k, v in uMap.iteritems():
    revUMap[v] = k
  with open(opFName, 'w') as g:
    for u in range(len(uMap.keys())):
      for item, rating in uiRatings[revUMap[u]].iteritems():
        if item in iMap: #as some items may not be present in filtSet
          g.write(str(iMap[item]) + ' ' + str(rating) + ' ')
      g.write('\n')


def getSetRatings(setFName):
  uSetRatings = {}
  nSets = 0
  with open(setFName, 'r') as f:
    for line in f:
      cols   = line.strip().split(',')
      user   = int(cols[0])
      uItems = map(int, cols[1].split('-'))
      rating = float(cols[2])
      if user not in uSetRatings:
        uSetRatings[user] = []
      uSetRatings[user].append((uItems, rating))
      nSets += 1
  print 'No. of sets: ', nSets
  return uSetRatings


def writeSetRating(user, setRatings, g, uMap, iMap):
  uItems = set([])
  for setRating in setRatings:
    for item in setRating[0]:
      uItems.add(item)
  g.write(str(uMap[user]) + ' ' + str(len(setRatings)) + ' ' + str(len(uItems)) + ' ')
  for item in list(uItems):
    g.write(str(iMap[item]) + ' ')
  g.write('\n')
  for setRating in setRatings:
    g.write(str(setRating[1]) + ' ' + str(len(setRating[0])) + ' ')
    for item in list(setRating[0]):
      g.write(str(iMap[item]) + ' ')
    g.write('\n')
 

def toLFS(uSetRatings, opFName, uMap, iMap):
  with open(opFName, 'w') as g:
    users = uSetRatings.keys()
    users.sort()
    for user in users:
      setRatings = uSetRatings[user]
      writeSetRating(user, setRatings, g, uMap, iMap)


def toCSR(uSetRatings, opFName, uMap, iMap, uiRatings):
  revUMap = {}
  for k, v in uMap.iteritems():
    revUMap[v] = k
  with open(opFName, 'w') as g:
    for u in range(len(uMap.keys())):
      uItems = set([])
      setRatings = []
      if revUMap[u] in uSetRatings:
        setRatings = uSetRatings[revUMap[u]]
      for setRating in setRatings:
        for item in setRating[0]:
          uItems.add(item)
      uItemsRat = []
      for item in list(uItems):
        uItemsRat.append((iMap[item], uiRatings[revUMap[u]][item]))
      uItemsRat.sort()
      for (item, rating) in uItemsRat:
        g.write(str(item) + ' ' + str(rating) + ' ')
      g.write('\n')


def toLFSTrainTestVal(uSetRatings, opCombName, opTrainName, opTestName, 
    opValName, uMap, iMap, uiRatings):
  invalUsers       = []
  combUSetRatings  = {}
  trainUSetRatings = {}
  testUSetRatings  = {}
  valUSetRatings   = {}
  nFiltSet         = 0
  with open(opTrainName, 'w') as tr, open(opTestName, 'w') as te, \
      open(opValName, 'w') as va, open(opCombName, 'w') as comb, \
      open("ml_set.und.lfs", "w") as und, open("ml_set.ovr.lfs", "w") as ovr:
    for user, setRatings in uSetRatings.iteritems():
      filtSetRatings = []
      undSetRatings = []
      ovrSetRatings = []
      for setRating in setRatings:
        items         = setRating[0]
        rating        = setRating[1]
        setRatingType = isSetUnderOverRat(user, items, rating, uiRatings)
        
        if (setRatingType == NEITHER_RATED_SET):
          filtSetRatings.append(setRating)
          nFiltSet += 1
        elif (setRatingType == UNDER_RATED_SET):
          undSetRatings.append(setRating)
        else:
          ovrSetRatings.append(setRating)

      if len(filtSetRatings) <= 2:
        invalUsers.append(user)
        continue
      random.shuffle(filtSetRatings)
      
      combUSetRatings[user] = filtSetRatings
      writeSetRating(user, filtSetRatings, comb, uMap, iMap)
      if len(undSetRatings) > 0:
        writeSetRating(user, undSetRatings, und, uMap, iMap)
      if len(ovrSetRatings) > 0:
        writeSetRating(user, ovrSetRatings, ovr, uMap, iMap)

      trainUSetRatings[user] = filtSetRatings[:-2]
      writeSetRating(user, filtSetRatings[:-2], tr, uMap, iMap)
      
      testUSetRatings[user] = filtSetRatings[-2:-1]
      writeSetRating(user, filtSetRatings[-2:-1], te, uMap, iMap)

      valUSetRatings[user] = filtSetRatings[-1:]
      writeSetRating(user, filtSetRatings[-1:], va, uMap, iMap)

  toCSR(combUSetRatings, "combine.csr", uMap, iMap, uiRatings)
  toCSR(trainUSetRatings, "train.csr", uMap, iMap, uiRatings)
  toCSR(testUSetRatings, "test.csr", uMap, iMap, uiRatings)
  toCSR(valUSetRatings, "val.csr", uMap, iMap, uiRatings)

  print 'No. of users not split: ', len(invalUsers)
  print 'No. of sets: ', nFiltSet


def toLFSTrainTestValOnly(uSetRatings, opCombName, opTrainName, opTestName, 
    opValName, uMap, iMap, uiRatings):
  
  invalUsers       = []
  combUSetRatings  = {}
  trainUSetRatings = {}
  testUSetRatings  = {}
  valUSetRatings   = {}
  nSet             = 0

  with open(opTrainName, 'w') as tr, open(opTestName, 'w') as te, \
      open(opValName, 'w') as va, open(opCombName, 'w') as comb:
    for user, setRatings in uSetRatings.iteritems():
      for setRating in setRatings:
        items         = setRating[0]
        rating        = setRating[1]
      
      if len(setRatings) < 5:
        invalUsers.append(user)
        continue

      nSet += 1
      random.shuffle(setRatings)
      
      combUSetRatings[user] = setRatings
      writeSetRating(user, setRatings, comb, uMap, iMap)

      trainUSetRatings[user] = setRatings[:-4]
      writeSetRating(user, setRatings[:-4], tr, uMap, iMap)
      
      testUSetRatings[user] = setRatings[-4:-2]
      writeSetRating(user, setRatings[-4:-2], te, uMap, iMap)

      valUSetRatings[user] = setRatings[-2:]
      writeSetRating(user, setRatings[-2:], va, uMap, iMap)

  toCSR(combUSetRatings, "combine.csr", uMap, iMap, uiRatings)
  toCSR(trainUSetRatings, "train.csr", uMap, iMap, uiRatings)
  toCSR(testUSetRatings, "test.csr", uMap, iMap, uiRatings)
  toCSR(valUSetRatings, "val.csr", uMap, iMap, uiRatings)

  print 'No. of users not split: ', len(invalUsers)
  print 'No. of sets: ', nFiltSet


""" neither as test sets"""
def toLFSTrainTestVal2(uSetRatings, opPrefix, uMap, iMap, uiRatings):
  invalUsers       = []
  combUSetRatings  = {}

  trainUSetRatings = {}
  testUSetRatings  = {}
  valUSetRatings   = {}
  
  nTrainUSetRatings = {}
  nValUSetRatings = {}
  
  nFiltSet         = 0
  
  opCombName  = opPrefix + '_set.comb.lfs'
  opTrainName = opPrefix + '_set.train.lfs'
  opTestName  = opPrefix + '_set.test.lfs'
  opValName   = opPrefix + '_set.val.lfs'

  nOpCombName  = opPrefix + '_set.comb.n.lfs'
  nOpTrainName = opPrefix + '_set.train.n.lfs'
  #nOpTestName  = opPrefix + '.test.n.lfs'
  nOpValName   = opPrefix + '_set.val.n.lfs'

  with open(opTrainName, 'w') as tr, open(opTestName, 'w') as te, \
      open(opValName, 'w') as va, open(opCombName, 'w') as comb, \
      open("ml_set.und.lfs", "w") as und, open("ml_set.ovr.lfs", "w") as ovr, \
      open(nOpTrainName, 'w') as ntr, open(nOpValName, 'w') as nva, \
      open(nOpCombName, 'w') as ncomb:
    for user, setRatings in uSetRatings.iteritems():
      neitherSetRatings = []
      undSetRatings = []
      ovrSetRatings = []
      for setRating in setRatings:
        items = setRating[0]
        rating  = setRating[1]
        setRatingType = isSetUnderOverRat(user, items, rating, uiRatings)
        if (setRatingType == NEITHER_RATED_SET):
          neitherSetRatings.append(setRating)
          nFiltSet += 1
        elif (setRatingType == UNDER_RATED_SET):
          undSetRatings.append(setRating)
        else:
          ovrSetRatings.append(setRating)

      if len(neitherSetRatings) <= 2:
        invalUsers.append(user)
        continue
      
      random.shuffle(neitherSetRatings)
      
      #write train val and test from neither
      writeSetRating(user, neitherSetRatings[:-2], ntr, uMap, iMap)
      writeSetRating(user, neitherSetRatings[-2:-1], nva, uMap, iMap)
      writeSetRating(user, neitherSetRatings[-1:], te, uMap, iMap)

      #write train val for all
      allSetRatings = ovrSetRatings + undSetRatings + neitherSetRatings[:-1]
      random.shuffle(allSetRatings)
      writeSetRating(user, allSetRatings[:-1], tr, uMap, iMap)
      writeSetRating(user, allSetRatings[-1:], va, uMap, iMap)

      if len(undSetRatings) > 0:
        writeSetRating(user, undSetRatings, und, uMap, iMap)
      if len(ovrSetRatings) > 0:
        writeSetRating(user, ovrSetRatings, ovr, uMap, iMap)

      trainUSetRatings[user] = allSetRatings[:-1]
      testUSetRatings[user] = neitherSetRatings[-1:]
      valUSetRatings[user] = allSetRatings[-1:]

      nTrainUSetRatings[user] = neitherSetRatings[:-2]
      nValUSetRatings[user] = neitherSetRatings[-2:-1]

  toCSR(trainUSetRatings, "train.csr", uMap, iMap, uiRatings)
  toCSR(testUSetRatings, "test.csr", uMap, iMap, uiRatings)
  toCSR(valUSetRatings, "val.csr", uMap, iMap, uiRatings)

  toCSR(nTrainUSetRatings, "train.n.csr", uMap, iMap, uiRatings)
  toCSR(nValUSetRatings, "val.n.csr", uMap, iMap, uiRatings)
  
  print 'No. of users not split: ', len(invalUsers)
  print 'No. of sets: ', nFiltSet



def ratingsNotInUSets(uiRatings, uSetRatings, notOpFName, inTrainFName,
    uMap, iMap):
  
  revUMap = {}
  for k, v in uMap.iteritems():
    revUMap[v] = k
  with open(notOpFName, 'w') as g, open(inTrainFName, 'w') as h:
    for u in range(len(uMap)):
      revU = revUMap[u]
      setRatings = uSetRatings[revU]
      setItems = set([])
      for setRating in setRatings:
        for item in setRating[0]:
          setItems.add(item)
      itemRatings = uiRatings[revU]
      for item, rating in itemRatings.iteritems():
        if item not in setItems and item in iMap:
          #write item and rating
          g.write(str(iMap[item]) + ' ' + str(rating) + ' ')
        elif item in setItems and item in iMap:
          #write item nd rating
          h.write(str(iMap[item]) + ' ' + str(rating) + ' ')
      g.write('\n')
      h.write('\n')


def main():
  ipSetFName = sys.argv[1]
  ipRatName  = sys.argv[2]
  opPrefix   = sys.argv[3]
  seed       = int(sys.argv[4])

  random.seed(seed)

  uSetRatings    = getSetRatings(ipSetFName)
  (users, items) = getUItems(ipSetFName)
  uiRatings      = getUIRatings(ipRatName)
  checkRatingsExist(uSetRatings, uiRatings)

  uMap = getMap(list(users))
  writeMap(uMap, opPrefix + '_uMap.txt')
  
  iMap = getMap(list(items))
  writeMap(iMap, opPrefix + '_iMap.txt')
  
  toLFS(uSetRatings, opPrefix + "_set.lfs", uMap, iMap)
  toLFSTrainTestValOnly(uSetRatings, opPrefix,  uMap, iMap, uiRatings)

  writeUIRatingsCSR(uiRatings, opPrefix + "_ratings.csr", uMap, iMap)
  ratingsNotInUSets(uiRatings, uSetRatings, 
      opPrefix + "_notTrainSet.csr",
      opPrefix + "_inTrainSet.csr",
      uMap, iMap)


if __name__ == '__main__':
  main()


