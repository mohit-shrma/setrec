import sys
import numpy as np


def getUIRatings(ratingsCSRFName):
  uiRatings = {}
  user = 0
  with open(ratingsCSRFName, 'r') as f:
    for line in f:
      cols = line.strip().split()
      for i in range(0, len(cols), 2):
        item = int(cols[i])
        rating = float(cols[i+1])
        if user not in uiRatings:
          uiRatings[user] = {}
        uiRatings[user][item] = rating
      user += 1
  return uiRatings


def getUserSets(setFName, uiRatings):
  uSets = {}
  with open(setFName, 'r') as f:
    line = f.readline()
    while (len(line) > 0):
      cols = line.strip().split()
      user = int(cols[0])
      nSets = int (cols[1])
      uSets[user] = []
      for i in range(nSets):
        setLine = f.readline()
        cols = setLine.strip().split()
        setRating = float(cols[0])
        itemsSet = []
        for item in cols[2:]:
          item = int(item)
          itemsSet.append(item)
          if user not in uiRatings:
            print 'ratings not found for', user
            print line
            print setLine
          elif item not in uiRatings[user]:
            print item, ' not rated by ', user
        uSets[user].append([setRating, itemsSet])
      line = f.readline()
  return uSets


def getSetRatingSim(user, itemsSet, uiRatings):
  setSz = len(itemsSet)
  sim = 0.0
  for i in range(setSz):
    itemI = itemsSet[i]
    for j in range(i+1, setSz):
      itemJ = itemsSet[j]
      diff  = uiRatings[user][itemI] - uiRatings[user][itemJ]
      sim   += diff*diff
  sim = np.sqrt(sim/((setSz*(setSz-1))/2))
  return sim


def getSetEntropy(user, itemsSet, uiRatings):
  setSz = len(itemsSet)
  #bins for rating: [0,1), [1,2), [2,3), [3,4), [4,5)
  bins = [0.0 for i in range(5)]  
  for item in itemsSet:
    rating = uiRatings[user][item]
    if rating < 1:
      bins[0] += 1
    elif rating < 2:
      bins[1] += 1
    elif rating < 3:
      bins[2] += 1
    elif rating < 4:
      bins[3] += 1
    else:
      bins[4] += 1

  entropy = 0.0
  for count in bins:
    if count > 0:
      entropy += -(count/setSz)*np.log10(count/setSz)
  return entropy


def getSetHiLoEntropy(user, itemsSet, uiRatings):
  setSz = len(itemsSet)
  #bins for rating: [0,3), [3,5]
  bins = [0.0 for i in range(2)]  
  for item in itemsSet:
    rating = uiRatings[user][item]
    if rating < 3:
      bins[0] += 1
    else:
      bins[1] += 1

  entropy = 0.0
  for count in bins:
    if count > 0:
      entropy += -(count/setSz)*np.log10(count/setSz)
  return entropy


def writeUserSetsWSimEntropy(uSets, uiRatings):
  for user, sets in uSets.iteritems():
    for [setRating, itemsSet] in sets:
      entropy     = getSetEntropy(user, itemsSet, uiRatings)
      hiLoEntropy = getSetHiLoEntropy(user, itemsSet, uiRatings)
      sim         = getSetRatingSim(user, itemsSet, uiRatings)
      itemRatings = []
      for item in itemsSet:
        itemRatings.append(uiRatings[user][item])
      print user, ' '.join(map(str, itemsSet)), ' '.join(map(str, itemRatings)), \
        sim, entropy, hiLoEntropy


def main():
  setFName        = sys.argv[1]
  ratingsCSRFName = sys.argv[2]
 
  uiRatings = getUIRatings(ratingsCSRFName)
  uSets = getUserSets(setFName, uiRatings)
  
  #TODO: write sets with entropy and diversity score
  writeUserSetsWSimEntropy(uSets, uiRatings)


if __name__ == '__main__':
  main()



