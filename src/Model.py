import numpy as np
import random

class Model:
  
  def __init__(self, nUsers, nItems, facDim, regU, regI, learnRate, maxIter,
      useSim = False):
    self.nUsers    = nUsers
    self.nItems    = nItems
    self.facDim    = facDim
    self.regU      = regU
    self.regI      = regI
    self.learnRate = learnRate
    self.maxIter   = maxIter
    self.useSim    = useSim
    self.uFactors = np.random.normal(size=(self.nUsers, self.facDim))
    #self.uFactors = np.ones((self.nUsers, self.facDim))
    self.iFactors = np.random.normal(size=(self.nItems, self.facDim))
    #self.iFactors = np.ones((self.nItems, self.facDim))
    

  def updateSim(self, sim):
    for i in range(self.nItems):
      sim[i][i] = 1.0
      for j in range(i+1, self.nItems):
        sim[i][j] = np.dot(self.iFactors[i], self.iFactors[j])
        sim[j][i] = sim[i][j]

  def saveLatFac(self):
    with open("uFactors.txt", "w") as g:
      for u in range(self.nUsers):
        for i in range(self.facDim):
          g.write(str(self.uFactors[u][i]) + ' ' )
        g.write('\n')
    with open("iFactors.txt", "w") as g:
      for u in range(self.nItems):
        for i in range(self.facDim):
          g.write(str(self.iFactors[u][i]) + ' ' )
        g.write('\n')
    


  def objective(self, arrUserSets):
    obj     = 0.0
    regUErr = 0.0
    regIErr = 0.0
    rmse    = 0.0
    
    for userSet in arrUserSets:
      user = userSet.user
      for i in range(len(userSet.items)):
        item = userSet.items[i]
        if item in userSet.testValItems:
          #item only in test or val set
          continue
        diff = userSet.itemsWt[i] - np.dot(self.uFactors[user], self.iFactors[item])
        rmse += diff*diff
        #print user, item, np.dot(self.uFactors[user], self.iFactors[item])
      regUErr += self.regU*np.dot(self.uFactors[user], self.uFactors[user])
    
    for item in range(self.nItems):
      regIErr += self.regI*np.dot(self.iFactors[item], self.iFactors[item])
    
    obj = regUErr + regIErr + rmse

    print 'Obj: ', obj, ' rmse: ', rmse, ' regU: ', regUErr, ' regI: ', regIErr
    return obj


  def train(self, arrUserSets):
    sim = np.zeros((self.nItems, self.nItems))
    for it in range(self.maxIter):
      
      for u in range(self.nUsers):
        userSet = arrUserSets[u]
        if it == 0:
          i = 0
        else:
          i = random.randint(0, len(userSet.items)-1)
        item = userSet.items[i]
        
        if item in userSet.testValItems:
          #item only in test or val set
          continue  

        uTv = np.dot(self.uFactors[u], self.iFactors[item])
        Wui = userSet.itemsWt[i]
        diff = Wui - uTv
        
        if item == 10:
          print item, Wui, uTv
        
        self.uFactors[u] += self.learnRate*(diff*self.iFactors[item]
            -self.regU*self.uFactors[u])
        self.iFactors[item] += self.learnRate*(diff*self.uFactors[u]
            -self.regI*self.iFactors[item]) 
        
      if self.useSim:
        #update similarities
        self.updateSim(sim)
        #TODO: update score

      if it%2 == 0:
        #objective
        self.objective(arrUserSets)
        #TODO: validation
