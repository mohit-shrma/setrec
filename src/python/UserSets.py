import random
import numpy as np

class UserSets:

  SZ_TEST = 1
  SZ_VAL = 1

  def __init__(self, user, numSets, items):
    self.user        = user
    self.numSets     = numSets
    self.itemSets    = []
    self.labels      = []
    self.testSetInds = []
    self.valSetInds  = []

    #items which only occur in test and val set
    self.testValItems = []
    
    #uniq items
    self.items = items
    
    #uniq items score
    self.itemsWt = [0.0 for i in range(len(items))]
    
    #mapping from item to sets
    self.item2Sets = {}

    #add val set ind
    for i in range(self.SZ_VAL):
      self.valSetInds.append(random.randint(0, self.numSets-1))

    #add test set ind
    j = 0
    while (j < self.SZ_TEST):
      ind = random.randint(0, self.numSets-1)
      if ind not in self.valSetInds:
        self.testSetInds.append(ind)
        j += 1


  def setSim(self, setInd, iFactors):
    itemSet = self.itemSets[setInd]
    nPairs = 0
    avgSimSet = 0.0
    for i in range(len(itemSet)):
      for j in range(i+1, len(itemSet)):
        avgSimSet += np.dot(iFactors[itemSet[i]],
            iFactors[itemSet[j]])
        nPairs += 1
    avgSimSet = avgSimSet/nPairs
    return avgSimSet 

  def addSet(self, itemSet, label):
    self.itemSets.append(itemSet)
    self.labels.append(float(label))
    for item in itemSet:
      if item not in self.item2Sets:
        self.item2Sets[item] = []
      self.item2Sets[item].append(len(self.itemSets)-1)
      
  
  def initWt(self):
    for i in range(len(self.items)):
      item = self.items[i]
      nSet = 0
      self.itemsWt[i] = 0
      for setInd in self.item2Sets[item]:
        if setInd in self.testSetInds or setInd in self.valSetInds:
          continue
        nSet += 1
        self.itemsWt[i] += float(self.labels[setInd])/len(self.itemSets[setInd]) 
      
      if nSet > 0:
        self.itemsWt[i] = self.itemsWt[i]/nSet
      else:
        self.itemsWt[i] = 0
        self.testValItems.append(item) 

  
  """
    rating for an item is same as that of set
    W_ui = sum(rating of sets with i)/(No. of sets with i)
  """
  def baseline1Scores(self):
    scores = {}
    
    for item in self.items:
      scores[item] = 0.0
      for setInd in self.item2Sets[item]:
        scores[item] += self.labels[setInd]
      scores[item] = scores[item]/len(self.item2Sets[item])

    return scores


  def baseline1ItemScore(self, item):
    score = 0.0
    for setInd in self.item2Sets[item]:
      score += self.labels[setInd]
    score = score/len(self.item2Sets[item])
    return score


  def baseline1SetScore(self, itemsSet):
    score = 0.0
    for item in itemsSet:
      score += self.baseline1ItemScore(item)
    score = score/len(itemsSet)
    return score

  
  def baseline1TestSetScore(self):
    scores = []
    for setInd in self.testSetInds:
      scores.append(self.baseline1SetScore(self.itemSets[setInd]))
    return scores


  """
    rating for the set is divided equally among the items in set
    W_ui = 1/(no. of sets with item i) * sum (rating for set / no. of items in
    set)
  """
  def baseline2Scores(self):
    scores = {}
    
    for item in self.items:
      scores[item] = 0.0
      for setInd in self.item2Sets[item]:
        scores[item] += self.labels[setInd]/len(self.itemSets[setInd])
      scores[item] = scores[item]/len(len(self.item2Sets[item]))

    return scores


  def baseline2ItemScore(self, item):
    score = 0.0
    for setInd in self.item2Sets[item]:
      score += self.labels[setInd]/len(self.itemSets[setInd])
    score = score/len(self.item2Sets[item])
    return score 


  def baseline2SetScore(self, itemsSet):
    score = 0.0
    for item in itemsSet:
      score += self.baseline2ItemScore(item)
    return score


  def baseline2TestSetScore(self):
    scores = []
    for setInd in self.testSetInds:
      scores.append(self.baseline2SetScore(self.itemSets[setInd]))
    return scores
 

  def testLabels(self):
    labels = []
    for setInd in self.testSetInds:
      labels.append(self.labels[setInd])
    return labels


  def dispWt(self):
    for i in range(len(self.items)):
      print self.items[i], self.itemsWt[i]

  #TODO
  def updWt(self):
    pass




