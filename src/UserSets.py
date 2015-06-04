import random

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


  def addSet(self, itemSet, label):
    self.itemSets.append(itemSet)
    self.labels.append(label)
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


  def dispWt(self):
    for i in range(len(self.items)):
      print self.items[i], self.itemsWt[i]


  def updWt(self):
    pass




