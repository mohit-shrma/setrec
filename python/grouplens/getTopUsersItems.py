import sys
import os

def getItemInSets(iMapFName):
  setItems = set([])
  iMap = {}
  with open(iMapFName, 'r') as f:
    for line in f:
      cols = line.strip().split(',')
      setItems.add(int(cols[0]))
      iMap[int(cols[0])] = int(cols[1])    
  return (setItems, iMap)


def getUItemCount(ipRatFName, setItems, uMapFName):
  
  exclUsers = set([])
  with open(uMapFName, 'r') as f:
    for line in f:
      cols = line.strip().split(',')
      exclUsers.add(cols[0]) 

  print 'no. of excluded users: ', len(exclUsers)
  print 'no. of set items: ', len(setItems) 
  uItemCount = {}
  f = open(ipRatFName, 'r')
  head = f.readline()
  for line in f:
    cols = line.strip().split(',')
    user = int(cols[0])
    item = int(cols[1])
    
    if user in exclUsers:      
      continue
    
    if item not in setItems:
        continue 

    if user not in uItemCount:
        uItemCount[user] = 0
    uItemCount[user] += 1
  f.close()
  
  print 'no. of users in uItemCount', len(uItemCount)

  return (uItemCount, exclUsers)


def writeTopUItemCount(uItemCount, fName):
  uItemCountTups = []
  for user, itemCount in uItemCount.iteritems():
    uItemCountTups.append((itemCount, user))
  uItemCountTups.sort(reverse=True)

  with open(fName, 'w') as g:
    for (itemCount, user) in uItemCountTups:
      g.write(str(user) + ',' + str(itemCount) + '\n')  
  
  return uItemCountTups


def getTopUsersRatings(uItemCountTups, nTopUsers, iMap, setItems, 
    ipRatFName, opRatFName):
  topUsers = set([])
  for (itemCount, user) in uItemCountTups[:nTopUsers]:
    topUsers.add(user)
  with open(ipRatFName, 'r') as f, open(opRatFName, 'w') as g:
    f.readline()
    for line in f:
      cols = line.strip().split(',')
      user = int(cols[0])
      item = int(cols[1])
      if item not in setItems:
        continue
      item = iMap[item] #rev mapped to orig ind
      rating = float(cols[2]) 
      if user in topUsers:
        g.write(str(user) + ' ' + str(item) + ' ' + str(rating) + '\n')


def getTopUsersRatingsCSR(uItemCountTups, nTopUsers, iMap, setItems, 
    ipRatFName, exclUsers, opRatFName):
  topUsers = set([])
  for (itemCount, user) in uItemCountTups[:nTopUsers]:
    topUsers.add(user)
  nRatings = 0
  with open(ipRatFName, 'r') as f, open(opRatFName, 'w') as g:
    f.readline()
    prevUser = ''
    for u in exclUsers:
        g.write('\n')
    for line in f:
      cols = line.strip().split(',')
      user = int(cols[0])
      item = int(cols[1])
      if item not in setItems:
        continue
      item = iMap[item] #rev mapped to orig ind
      rating = float(cols[2]) 
      if user in topUsers:
        if prevUser != '' and prevUser != user:
          g.write('\n')
        g.write(str(item) + ' ' + str(rating) + ' ')
        nRatings += 1
        prevUser = user
    g.write('\n')    
  print 'No. of ratings: ', nRatings


def main():
  ipRatFName = sys.argv[1]
  iMapFName  = sys.argv[2]
  uMapFName  = sys.argv[3]
  nTopUsers  = int(sys.argv[4])
  topUIRatFName = sys.argv[5]

  (setItems, iMap) = getItemInSets(iMapFName)
  print 'no. of set items: ', len(setItems)

  (uItemCount, exclUsers) = getUItemCount(ipRatFName, setItems, uMapFName)

  uItemCountTups = writeTopUItemCount(uItemCount, 'topUItemsCount.txt') 
  getTopUsersRatingsCSR(uItemCountTups, nTopUsers, iMap, setItems, 
      ipRatFName, exclUsers, topUIRatFName)  

  
if __name__ == '__main__':
  main()


