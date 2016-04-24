import sys

itemUserFreq = {}
itemSetFreq = {}
itemTop50PcFreq = {}
nUsers = 1000

fName = sys.argv[1]

#assuming item in sets are in decreasing order of their rating
with open(fName, 'r') as f:
  
  u = 0

  while (u < 1000):
    uHead = f.readline()  
    cols = map(int, uHead.split())
    
    if u != cols[0]:
      print 'ERRR u '
    
    nSets  = cols[1]
    
    for item in cols[2:]:
      if item not in itemUserFreq:
        itemUserFreq[item] = 0.0
      itemUserFreq[item] += 1
    
    #go over sets
    for i in range(nSets):
      setStr = f.readline()
      setCols = setStr.split()
      setItems = map(int, setCols[2:])
      for k in range(len(setItems)):
        item = setItems[k]
        if item not in itemSetFreq:
          itemTop50PcFreq[item] = 0.0
          itemSetFreq[item] = 0.0
        itemSetFreq[item] += 1
        
        if k <= 2:
          itemTop50PcFreq[item] += 1

    u += 1
    

items = itemUserFreq.keys()
with open('itemFreq.txt', 'w') as g:
  for item in items:
    g.write(str(item) + ' ' + str(itemTop50PcFreq[item]) + ' ' \
        + str(itemSetFreq[item]) +  ' ' \
        +  '%.3f' % (itemTop50PcFreq[item]/itemSetFreq[item])  + ' ' \
        +   str(itemUserFreq[item]) + '\n')



