import sys
import random



def getUserItems(ratFileName):
  userItemsRat = {}
  with open(ratFileName, 'r') as f:
    for line in f:
      cols = line.strip().split(',')
      user = int(cols[0])
      item = int(cols[1])
      rat = float(cols[2])
      if user not in userItemsRat:
        userItemsRat[user] = {}
      userItemsRat[user][item] = rat
  return userItemsRat


def getSetsForUser(itemRats, nSetsPerUser, setSize, thresh):
  
  nItems = len(itemRats)
  setLabels = []

  if nItems < 1.5 * setSize:
    return setLabels
  
  items = itemRats.keys()

  for i in range(nSetsPerUser):
    #generate set i
    tempSet = set([])
    while len(tempSet) < setSize:
      item = items[random.randint(0, nItems-1)]
      tempSet.add(item)
    
    #assign set label
    setItemRat = [] 
    for item in tempSet:
      setItemRat.append((itemRats[item], item))
    setItemRat.sort()
    
    #get avg of top items
    sm = 0.0
    nTopItems = 0.0
    for i in range(setSize/2, nItems):
      sm += setItemRat[i][0]
      nTopItems += 1
    avgTopRatItems = sm/nTopItems
    
    label = 1.0
    if avgTopRatItems < thresh:
      label = -1.0

    setLabels.append((tempSet, label))

  return setLabels


def genSetsNWrite(userItemsRat, opFileName, setSize, thresh, nSetsPerUser):
  
  with open(opFileName, 'w') as g, open(opFileName + '_map', 'w') as h:
    u = 0
    for user, itemRats in userItemsRat.iteritems():
      setLabels = getSetsForUser(itemRats, nSetsPerUser, setSize, thresh)
      nSets = len(setLabels)
      setItems = set([])
      for (st, label) in setLabels:
        setItems = setItems | st
      g.write(str(u) + ' ' + str(nSets) + ' ' + str(len(setItems)) + '\n')
      for (st, label) in setLabels:
        g.write(str(label) + ' ' + str(len(st)) + ' ' 
            + ' '.join(map(str, list(st))) + '\n')
      h.write(str(u) + ' ' + str(user) + '\n')
      u += 1


def main():
  ratFileName = sys.argv[1]
  setSize = int(sys.argv[2])
  thresh = float(sys.argv[3])
  nSetsPerUser = int(sys.argv[4])
  opFileName = sys.argv[5]

  userItemsRat = getUserItems(ratFileName)
  genSetsNWrite(userItemsRat, opFileName, setSize, thresh, nSetsPerUser)

if __init__ == '__main__':
  main()
