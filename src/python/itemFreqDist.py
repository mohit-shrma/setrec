itemUserFreq = {}
itemSetFreq = {}
nUsers = 1000

with open('simulMajSet_split1', 'r') as f:
  u = 0

  while (u < 1000):
    uHead = f.readline()  
    cols = map(int, uHead.split())
    
    if u != cols[0]:
      print 'ERRR u '
    
    nSets  = cols[1]
    
    for item in cols[1:]:
      if item not in itemUserFreq:
        itemUserFreq[item] = 0
      itemUserFreq[item] += 1
    
    #go over sets
    for i in range(nSets):
      setStr = f.readline()
      setCols = setStr.split()
      setItems = map(int, setCols[2:])
      for item in setItems:
        if item not in itemSetFreq:
          itemSetFreq[item] = 0
        itemSetFreq[item] += 1
        
    u += 1
    




