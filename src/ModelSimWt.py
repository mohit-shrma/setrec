import numpy as np
import random
from Model import Model

class ModelSimWt(Model):
  

  def itemGrad(self, userSet, setInd, item, avgSimSet):
    user = userSet.user
    itemSet = userSet.itemSets[setInd]
    r_us = userSet.labels[setInd]
    uFac = self.uFactors[user]
    iFac = self.iFactors[item]
    nItems = len(itemSet)

    #sum item lat fac in set
    sumItemLatFac = np.zeros(self.facDim)
    for itm in itemSet:
      sumItemLatFac += self.iFactors[itm]

    nPairs = (nItems * (nItems-1))/2
    comDiff = r_us - (1.0/len(itemSet))*np.dot(uFac, sumItemLatFac)
    grad = 2.0 * comDiff * (-1.0/len(itemSet)) * uFac *avgSimSet
    grad += (comDiff**2)* (1.0/nPairs)*(sumItemLatFac - iFac)
    
    return grad


  def userGrad(self, userSet, setInd, avgSimSet):
    user    = userSet.user
    itemSet = userSet.itemSets[setInd]
    r_us    = userSet.labels[setInd]
    uFac    = self.uFactors[user]
    
    #sum item lat fac in set
    sumItemLatFac = np.zeros(self.facDim)
    for item in itemSet: 
      sumItemLatFac += self.iFactors[item]

    grad = 2.0 * (r_us - ((1.0/len(itemSet))*np.dot(uFac, sumItemLatFac)))
    grad = grad* ((-1.0/len(itemSet)) * sumItemLatFac) 
    grad = grad * avgSimSet

    return grad


  def userSetLossI(self, userSet, setInd, item, iFac = None):
    
    user    = userSet.user
    itemSet = userSet.itemSets[setInd]
    r_us    = userSet.labels[setInd]
    uFac    = self.uFactors[user]

    if iFac is None:
      iFac = self.iFactors[item]

    #sum item lat fac in set
    sumItemLatFac = np.zeros(self.facDim)
    for itm in itemSet:
      if itm != item:
        sumItemLatFac += self.iFactors[itm]
      else:
        sumItemLatFac += iFac

    avgSimSet = 0.0
    nPairs = 0
    for i in range(len(itemSet)):
      for j in range(i+1, len(itemSet)):
        nPairs += 1
        if itemSet[i] == item:
          avgSimSet += np.dot(iFac, self.iFactors[itemSet[j]])
        elif itemSet[j] == item:
          avgSimSet += np.dot(self.iFactors[itemSet[i]], iFac)
        else:
          avgSimSet += np.dot(self.iFactors[itemSet[i]],
              self.iFactors[itemSet[j]])
    avgSimSet = avgSimSet/nPairs

    loss = ((r_us - ((1.0/len(itemSet))*np.dot(uFac, sumItemLatFac)))**2.0) * avgSimSet
    
    return loss


  def userSetLossU(self, userSet, setInd, avgSimSet, uFac = None):
    
    user    = userSet.user
    itemSet = userSet.itemSets[setInd]
    r_us    = userSet.labels[setInd]
    
    if uFac is None:
      uFac = self.uFactors[userSet.user]

    #sum item lat fac in set
    sumItemLatFac = np.zeros(self.facDim)
    for item in itemSet:
      sumItemLatFac += self.iFactors[item]

    loss = ((r_us - ((1.0/len(itemSet))*np.dot(uFac, sumItemLatFac)))**2.0) * avgSimSet
    
    return loss


  def gradCheck(self, userSet):
    
    #check gradient w.r.t. user
    print 'user gradient checks'
    setInd = random.randint(0, userSet.numSets - 1) 
    avgSimSet = userSet.setSim(setInd, self.iFactors)
    uGrad = self.userGrad(userSet, setInd, avgSimSet)
    
    uFac = self.uFactors[userSet.user]
    userSetLoss = self.userSetLossU(userSet, setInd, avgSimSet, uFac)

    """
    for i in range(5):
      delta = np.random.normal(0, 0.0001, size=self.facDim) 
      uFacTmp = uFac + delta 
      userSetLossTmp = self.userSetLossU(userSet, setInd, avgSimSet, uFacTmp)
      uFacTmp2 = uFac - delta
      userSetLossTmp2 = self.userSetLossU(userSet, setInd, avgSimSet, uFacTmp2) 
      diff = userSetLossTmp - userSetLoss - np.dot(delta, uGrad)
      div = ((userSetLossTmp - userSetLossTmp2) / 2.0) / (np.dot(delta, uGrad))
      print 'diff: ', diff, ' div: ', div
    """

    #gradient check for randomly selected item
    print 'item gradient checks'
    itemInd = random.randint(0, len(userSet.itemSets[setInd])-1)
    item = userSet.itemSets[setInd][itemInd]
    
    iGrad = self.itemGrad(userSet, setInd, item, avgSimSet)
    iFac = self.iFactors[item]
    for i in range(5):
      delta = np.random.normal(0, 0.0001, size=self.facDim) 
      iFacTmp = iFac + delta 
      userSetLossTmp = self.userSetLossI(userSet, setInd, item, iFacTmp)
      iFacTmp2 = iFac - delta
      userSetLossTmp2 = self.userSetLossI(userSet, setInd, item, iFacTmp2)
      diff = userSetLossTmp - userSetLoss - np.dot(delta, iGrad)
      div = ((userSetLossTmp - userSetLossTmp2) / 2.0) / (np.dot(delta, iGrad))
      print 'diff: ', diff, ' div: ', div



    


