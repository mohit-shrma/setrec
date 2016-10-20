import sys
import numpy as np


def getUISparsity(ratFName):
  usersItems = []
  allItems = set([])
  nUsers = 0
  nRatings = 0
  print 'Reading...', ratFName
  with open(ratFName, 'r') as f:
    for line in f:
      cols = line.strip().split()
      items = [ int(cols[i]) for i in range(0, len(cols), 2) ]
      items.sort()
      usersItems.append(items)
      for item in items:
        allItems.add(item)
      nRatings += len(items)
      nUsers += 1
  print 'nUsers: ', nUsers, 'nItems: ', len(allItems), 'nRatings: ', nRatings
  return (usersItems, nUsers, len(allItems))


def writeCSR(uItems, uFacs, iFacs, opFileName):
  with open(opFileName, 'w') as g:
    for u in range(len(uItems)):
      for item in uItems[u]:
        g.write(str(item) + ' ' + str(itemRating(uFacs, iFacs, u, item)) + ' ')
      g.write('\n')


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
      #a = np.random.rand(nChoices)
      #pDist = a/sum(a)
      a = np.zeros(nChoices)
      a[np.random.choice(nChoices)] = 1
      pDist = a
      pickyDist[u] = pDist
      g.write(' '.join(map(str, pDist)) + '\n')
  return pickyDist


def selectChoiceFromDist(dist):
  choice = np.random.choice(np.arange(len(dist)), p = dist)
  return choice


def setRatingByChoice(sortedRatings, choice):
  setRating = addNoise()
  if choice == 0:
     setRating += sortedRatings[0]
  elif choice == 1:
    setRating += sum(sortedRatings[:2])/2
  elif choice == 2:
    setRating += sum(sortedRatings[:3])/3
  elif choice == 3:
    setRating += sum(sortedRatings[:4])/4
  elif choice == 4:
    setRating += sum(sortedRatings[:5])/5
  elif choice == 5:
    setRating += sum(sortedRatings[1:5])/4
  elif choice == 6:
    setRating += sum(sortedRatings[2:5])/3
  elif choice == 7:
    setRating += sum(sortedRatings[3:5])/2
  elif choice == 8:
    setRating += sortedRatings[4]
  else:
    print "ERROR: invalid choice found"
  return setRating


def addNoise(loc = 0, scale = 0.1):
  return np.random.normal(loc, scale)


def setWVarRating(user, items, uFacs, iFacs, uWeights, gamma = 0):
  ratings = []
  for item in items:
    ratings.append(np.dot(uFacs[user], iFacs[item]) + addNoise())
  ratings = np.asarray(ratings) 
  meanRat = np.mean(ratings)
  stdRat = np.std(ratings)
  return meanRat + uWeights[user]*(gamma + stdRat) + addNoise()


def setRating(user, items, uFacs, iFacs, pickyDist):
  ratings = []
  for item in items:
    ratings.append(np.dot(uFacs[user], iFacs[item]) + addNoise())
  ratings.sort()
  uChoice = selectChoiceFromDist(pickyDist[user])
  #return setRatingByChoice(ratings, 4) #average model 
  return setRatingByChoice(ratings, uChoice) 


def itemRating(uFacs, iFacs, user, item):
  return (np.dot(uFacs[user], iFacs[item]) + addNoise())


def genTrainTestValSets(uItemSets):
  uTrainItemSets = {}
  uTestItemSets   = {}
  uValItemSets    = {}
  nFound = 0
  for user, itemSets in uItemSets.iteritems():
    nSets = len(itemSets)
    if nSets < 15:
      nFound += 1
      continue
    uTrainItemSets[user] = []
    uTestItemSets[user] = []
    uValItemSets[user] = []
    testValSetsInd = set(np.random.choice(np.arange(nSets), 10, 
      replace=False))
    for ind in range(nSets):
      if ind not in testValSetsInd:
        uTrainItemSets[user].append(itemSets[ind])
    testValIndList = list(testValSetsInd)
    uTestItemSets[user] = [itemSets[ind] for ind in testValIndList[:5]]
    uValItemSets[user] = [itemSets[ind] for ind in testValIndList[5:]]
  print 'Ignored ... ', nFound, 'users'
  return [uTrainItemSets, uTestItemSets, uValItemSets]


def genSets(items):
  nTry = 0;
  itemSets = []
  while len(itemSets) < 50 and nTry < 1000:
    nTry += 1
    s = np.random.choice(items, 5, replace=False)
    isOverlap = False
    for prevSet in itemSets:
      if len(prevSet & set(s)) > 0:
        isOverlap = True
        break
    if not isOverlap:
      itemSets.append(set(s))
  return itemSets


def genSynSets(uItems):
  nUsers = len(uItems)
  arrUsers = np.arange(nUsers)
  np.random.shuffle(arrUsers)
  uItemSets = {}
  for user in arrUsers:
    #gen 50 sets from the rated item sparsity
    if len(uItems[user]) < 100:
      continue
    itemSets = genSets(uItems[user])
    if len(itemSets) < 20:
      continue
    uItemSets[user] = itemSets
    if len(uItemSets) == 1000:
      break
  print 'Generated sets for ', len(uItemSets), ' users'
  return uItemSets


def writeSet(opFile, user, itemSets, uFacs, iFacs, pickyDist, uWeights):
  unionItems = set([])
  for itemSet in itemSets:
    unionItems |= itemSet
  nItems = len(unionItems)
  items = list(unionItems)
  items.sort()
  
  #write header
  opFile.write(str(user) + ' ')
  opFile.write(str(len(itemSets)) + ' ')
  opFile.write(str(nItems) + ' ')
  for item in items:
    opFile.write(str(item) + ' ')
  opFile.write('\n')
  
  #write sets
  for itemSet in itemSets:
    #rating = setRating(user, itemSet, uFacs, iFacs, pickyDist)
    rating = setWVarRating(user, itemSet, uFacs, iFacs, uWeights)
    setSz = len(itemSet)
    opFile.write(str(rating) + ' ' + str(setSz) + ' ')
    for item in itemSet:
      opFile.write(str(item) + ' ')
    opFile.write('\n')


def writeSynSets(uItemSets, uFacs, iFacs, opFileName, pickyDist, uWeights=[]):
  sampUsers = uItemSets.keys()
  sampUsers.sort()
  #randomly select 1000 users
  with open(opFileName, 'w') as g:
    for user in sampUsers:
      #50 sets from the rated item sparsity
      itemSets = uItemSets[user]
      writeSet(g, user, itemSets, uFacs, iFacs, pickyDist, uWeights) 


def writeRatMat4Set(uItemSets, nUsers, uFacs, iFacs, opFileName):
  with open(opFileName, 'w') as g:
    for u in range(nUsers):
      if u not in uItemSets:
        g.write('\n')
        continue
      items = set([])
      for setItems in uItemSets[u]:
        for item in setItems:
          items.add(item)
      items = list(items)
      items.sort()
      for item in items:
        g.write(str(item) + ' ' + str(itemRating(uFacs, iFacs, u, item)) + ' ')
      g.write('\n')


def writeRatMat4UIPairs(uItemPairsD, nUsers, uFacs, iFacs, opFileName):
  with open(opFileName, 'w') as g:
    for u in range(nUsers):
      if u not in uItemPairsD:
        g.write('\n')
        continue
      for item in uItemPairsD[u]:
        g.write(str(item) + ' ' + str(itemRating(uFacs, iFacs, u, item)) + ' ')
      g.write('\n')

"""
0 - sample rating for users with items only items inside set
1 - sample ratings for users with items outside their sets
2 - sample ratings for users with items both inside and outside set
3 - sample ratings for users for items outside of all the sets
4 - sample ratings for items in the sets but for different users
"""
def genPartMat(uTrainItemSets, fullUItems, categ=0, sampPc=0.25):
  allItems = set([])  
  uTrItems = {}
  allTrItems = set([])
  allTrRatings = 0
  trUsers = []

  for itemSet in fullUItems:
    for item in itemSet:
      allItems.add(item)
  allItems = list(allItems)
  
  print 'all items: ', len(allItems)

  for user, trSets in uTrainItemSets.iteritems():
    uItems = set([])
    for s in trSets:
      for item in s:
        uItems.add(item)
        allTrRatings += 1
        allTrItems.add(item)
    uTrItems[user] = list(uItems)
    trUsers.append(user)
  
  allTrItems = list(allTrItems)
  
  print 'all Train items: ', len(allTrItems)

  itemsNotInTr = list(set(allItems) - set(allTrItems))
  
  print 'items not in train', len(itemsNotInTr)

  usersNotInTr = list(set(range(len(fullUItems))) - set(trUsers))
  
  print 'users not in train: ', len(usersNotInTr)

  sampUsersItem = {}
  uiPairCount = 0
  nTry = 0
  while uiPairCount < sampPc*allTrRatings \
    and nTry < sampPc*allTrRatings*10:
    sampPair = ()
    if categ == 0:
      #sample user
      u = np.random.choice(trUsers)
      #sample item 
      item = np.random.choice(uTrItems[u])
      sampPair = (u,item)
    elif categ == 1:
      #sample user
      u = np.random.choice(trUsers)
      #sample item 
      item = np.random.choice(allTrItems)
      if item not in uTrItems[u]:
        sampPair = (u,item)
    elif categ == 2:
      #sample user
      u = np.random.choice(trUsers)
      #sample item 
      item = np.random.choice(allTrItems)
      sampPair = (u,item)
    elif categ == 3:
      #sample user
      u = np.random.choice(trUsers)
      #sample item 
      item = np.random.choice(itemsNotInTr)
      sampPair = (u,item)
    elif categ == 4:
      #sample user
      u = np.random.choice(usersNotInTr)
      #sample item 
      item = np.random.choice(allTrItems)
      sampPair = (u,item)

    if len(sampPair) == 0:
      continue
    
    #check if sampled pair unique
    (u, item) = sampPair
    if u not in sampUsersItem:
      sampUsersItem[u] = set([])
    if item not in sampUsersItem[u]:
      uiPairCount += 1
      sampUsersItem[u].add(item)

    nTry += 1
 
  print 'No. of sampled ratings: ', uiPairCount
  print 'All train ratings: ', allTrRatings
  print 'Sampled %: ', float(uiPairCount)/float(allTrRatings)

  return sampUsersItem


def writeSynCSR(ipFileName, opFileName, nUsers, nItems, uFacs, iFacs):
  u = 0
  with open(ipFileName, 'r') as f, open(opFileName, 'w') as g:
    for line in f:
      cols = line.strip().split()
      if len(cols) > 0:
        for i in range(0, len(cols), 2):
          item = int(cols[i])
          g.write(str(item) + ' ' + str(np.dot(uFacs[u], iFacs[item]) +
            addNoise()) + ' ')
      g.write('\n')
      u += 1


def main():

  ipFName  = sys.argv[1] #ipCSR
  facDim   = int(sys.argv[2])
  scale    = int(sys.argv[3])
  randSeed = int(sys.argv[4])
  
  gamma  = 0.25 #TODO: whether to use in rating gen?
  if len(sys.argv) >  5:
    gamma = float(sys.argv[5])

  np.random.seed(randSeed)

  #read sparsity structure of the input rating matrix
  (fullUItems, nUsers, nItems) = getUISparsity(ipFName)

  #gen known low-rank model
  (uFacs, iFacs) = genScaledFacs(nUsers, nItems, facDim, scale)
  
  print 'Writing full rating matrix...'
  writeCSR(fullUItems, uFacs, iFacs, 'ratings.syn.csr')

  #gen pickiness dist
  print 'Generating pickiness...' 
  pickiDists = []
  for i in range(5):
    pickiDist = generateUserPickynessDist(nUsers, 
        "userPickiness_syn_" + str(i) + ".txt")
    pickiDists.append(pickiDist)

  #gen user weights from -2 to 2
  print 'Generating uweights...' 
  uMultWeights = []
  for i in range(5):
    uWeights = -2 + (2 - -2)*np.random.rand(nUsers);
    np.savetxt("uWeights_syn_" + str(i) + ".txt", uWeights)
    uMultWeights.append(uWeights)

  #generate sets for randomly sampled 1000 users
  print 'Generating sets...'
  uItemSets = genSynSets(fullUItems)
  [uTrainItemSets, uTestItemSets, uValItemSets] = genTrainTestValSets(uItemSets)
 
  #write  matrices for train, test and val sets
  print 'Saving individual rating matrices...'
  writeRatMat4Set(uItemSets, nUsers, uFacs, iFacs, "set_ratings.syn.csr",)
  writeRatMat4Set(uTrainItemSets, nUsers, uFacs, iFacs, "train.syn.csr",)
  writeRatMat4Set(uTestItemSets, nUsers, uFacs, iFacs, "test.syn.csr",)
  writeRatMat4Set(uValItemSets, nUsers, uFacs, iFacs, "val.syn.csr",)
    
  #gen sampled matrices
  print 'Categ 0 matrix...'
  samp_categ0_mat = genPartMat(uTrainItemSets, fullUItems, 0, 0.25) 
  writeRatMat4UIPairs(samp_categ0_mat, nUsers, uFacs, iFacs, 'train_0.syn.csr')
  
  print 'Categ 1 matrix...'
  samp_categ1_mat = genPartMat(uTrainItemSets, fullUItems, 1, 0.25) 
  writeRatMat4UIPairs(samp_categ1_mat, nUsers, uFacs, iFacs, 'train_1.syn.csr')

  print 'Categ 2 matrix...'
  samp_categ2_mat = genPartMat(uTrainItemSets, fullUItems, 2, 0.25) 
  writeRatMat4UIPairs(samp_categ2_mat, nUsers, uFacs, iFacs, 'train_2.syn.csr')
  
  print 'Categ 3 matrix...'
  samp_categ3_mat = genPartMat(uTrainItemSets, fullUItems, 3, 0.25) 
  writeRatMat4UIPairs(samp_categ3_mat, nUsers, uFacs, iFacs, 'train_3.syn.csr')
    
  print 'Categ 4 matrix...'
  samp_categ4_mat = genPartMat(uTrainItemSets, fullUItems, 4, 0.25) 
  writeRatMat4UIPairs(samp_categ4_mat, nUsers, uFacs, iFacs, 'train_4.syn.csr')

  #write syn sets
  print 'Saving sets...'
  for i in range(5):
    writeSynSets(uItemSets, uFacs, iFacs, "ml_set.syn_" + str(i) + ".lfs", 
        pickiDists[i], uMultWeights[i])
    writeSynSets(uTrainItemSets, uFacs, iFacs, "ml_set.train.syn_" + str(i) + ".lfs",
        pickiDists[i], uMultWeights[i])
    writeSynSets(uTestItemSets, uFacs, iFacs, "ml_set.test.syn_" + str(i) + ".lfs",
        pickiDists[i], uMultWeights[i])
    writeSynSets(uValItemSets, uFacs, iFacs, "ml_set.val.syn_" + str(i) + ".lfs",
        pickiDists[i], uMultWeights[i])
 

if __name__ == '__main__':
  main()


