import sys

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
    for user, setRatings in uSetRatings.iteritems():
      writeSetRating(user, setRatings, g, uMap, iMap)


def toLFSTrainTestVal(uSetRatings, opTrainName, opTestName, opValName, 
    uMap, iMap):
  invalUsers = []
  with open(opTrainName, 'w') as tr, open(opTestName, 'w') as te, open(opValName, 'w') as va:
    for user, setRatings in uSetRatings.iteritems():
      if len(setRatings) <= 2:
        invalUsers.append(user)
      writeSetRating(user, setRatings[:-2], tr, uMap, iMap)
      writeSetRating(user, setRatings[-2:-1], te, uMap, iMap)
      writeSetRating(user, setRatings[-1:], va, uMap, iMap)
  print 'No. of users not split: ', len(invalUsers)


def main():
  ipSetFName = sys.argv[1]
  ipRatName  = sys.argv[2]
  opPrefix    = sys.argv[3]
  
  uSetRatings    = getSetRatings(ipSetFName)
  (users, items) = getUItems(ipSetFName)
  uiRatings      = getUIRatings(ipRatName)
  checkRatingsExist(uSetRatings, uiRatings)

  uMap = getMap(list(users))
  writeMap(uMap, opPrefix + '_uMap.txt')
  
  iMap = getMap(list(items))
  writeMap(iMap, opPrefix + '_iMap.txt')
  
  toLFS(uSetRatings, opPrefix + "_set.lfs", uMap, iMap)
  toLFSTrainTestVal(uSetRatings, opPrefix + "_set.train.lfs", 
      opPrefix + "_set.test.lfs",
      opPrefix + "_set.val.lfs",
      uMap, iMap)

  writeUIRatingsCSR(uiRatings, opPrefix + "_ratings.csr", uMap, iMap)


if __name__ == '__main__':
  main()


