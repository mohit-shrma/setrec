import sys
import MySQLdb
import datetime

def getInvalidUsersWRat(uiRatings, uSetRatings):
  invalidUsers = set([])
  for user, itemsSetsRating in uSetRatings.iteritems():
    for itemSetRating in itemsSetsRating:
      items = itemSetRating[0]
      rating = itemSetRating[1]
      for item in items:
        if item not in uiRatings[user]:
          invalidUsers.add(user)
          break
      if user in invalidUsers:
        break
  print 'Ratings not found for: ', len(invalidUsers)
  return invalidUsers


def getInvalidUsersByTS(setFName):
  #order ratings for user in set by tstamp, then remove those users who rated
  #set with less than one sec bw sets, 
  #read sorted file by users
  invalidUsers = []
  with open(setFName, 'r') as f:
    prevUser = None
    uTStamps = []
    for line in f:
      cols = line.strip().split(',')
      currUser = int(cols[0])
      currUItems = map(int, cols[1].split('-'))
      currRat = float(cols[2])
      strTs = cols[3]
      currTs = datetime.datetime.strptime(strTs, "%Y-%m-%d %H:%M:%S")
      if prevUser is not None and prevUser != currUser:
        #found new user, analyze current user
        uTStamps.sort()
        #check if consecutive ts difference < 1000ms
        for i in range(len(uTStamps)-1):
          ts1 = uTStamps[i]
          ts2 = uTStamps[i+1]
          diff = ts2 - ts1
          diffSecs = diff.total_seconds()
          if diffSecs < 1:
            invalidUsers.append(currUser)
            break
        #reset for next user 
        uTStamps = []

      uTStamps.append(currTs)
      prevUser = currUser
    #analyze last user
    uTStamps.sort()
    #check if consecutive ts difference < 1000ms
    for i in range(len(uTStamps)-1):
      ts1 = uTStamps[i]
      ts2 = uTStamps[i+1]
      diff = ts2 - ts1
      diffSecs = diff.total_seconds()
      if diffSecs < 1:
        invalidUsers.append(currUser)
        break
  print 'no. of invalid users by TS: ', len(invalidUsers)
  return set(invalidUsers)  


def writeList(lst, opFName):
  with open(opFName, 'w') as g:
    for l in lst:
      g.write(str(l) + '\n')


def getDBCursor():
  db = MySQLdb.connect(host="flagon.cs.umn.edu",     # your host
                       user="readonly",              # your username
                       passwd="",                    # your password
                       db="ML3_mirror")              # name of the data base
  #get cursor to query
  cur = db.cursor()
  return cur


def filtSetRatings(uSetRatings, invalUsers, opFileName):
  nSetRatings = 0
  with open(opFileName, 'w') as g:
    for user, setRatings in uSetRatings.iteritems():
      if user in invalUsers:
        continue
      for itemSetRating in setRatings:
        itemSet = itemSetRating[0]
        setRating = itemSetRating[1]
        g.write(str(user) + ',' + '-'.join(map(str, itemSet)) + ',' +
             str(setRating) + '\n')
        nSetRatings += 1
  print 'No. set ratings after filter: ', nSetRatings
        

def getSetRatings(cur, opFileName):
  uSetRatings = {}
  nSetRatings = 0
  #query to get all the data
  cur.execute('select * from expt_set_user_rating order by userId, tstamp')
  with open(opFileName, 'w') as g:
    for row in cur.fetchall():
      user = int(row[1])
      setItems = map(int, row[2].split('-'))
      setRating = float(row[3])
      if user not in uSetRatings:
        uSetRatings[user] = []
      uSetRatings[user].append((setItems, setRating))
      nSetRatings += 1
      strRow = map(str, row)
      g.write(','.join(strRow[1:]) + '\n')
  print 'nSetRatings: ', nSetRatings
  return uSetRatings


def getUserItems(uSetCSVFName):
  users = set([])
  items = set([])
  with open(uSetCSVFName, 'r') as f:
    for line in f:
      cols = line.strip().split(',')
      user = int(cols[0])
      uItems = map(int, cols[1].split('-'))
      users.add(user)
      for uItem in uItems:
        items.add(uItem)
  return (users, items)


def getUserItemRatings(cur, users, items, opFileName):
  #query to get join of ratings and set_ratings
  cur.execute('select A.userId, A.movieId, A.rating, A.tstamp from \
      user_rating_pairs A INNER JOIN expt_set_user_rating B ON \
      A.userId=B.userId and A.rating != -1;')
  uIRat = {}
  nnz = 0
  with open(opFileName, 'w') as g:
    for row in cur.fetchall():
      user   = row[0]
      item   = row[1]
      rating = row[2]
      if user in users and item in items:
        if user not in uIRat:
          uIRat[user] = {}
        if item not in uIRat[user]:
          uIRat[user][item] = rating
          nnz += 1
          strRow = map(str, row)
          g.write(','.join(strRow) + '\n')
        elif uIRat[user][item] != rating:
          print 'Duplicate rating don\'t match: ', user, item
  print 'no. of uIRatings: ', nnz
  return uIRat


def main():
  opPrefix = sys.argv[1]
  cur = getDBCursor()
  
  setFName = opPrefix + '_set_ratings.csv'
  uSetRatings = getSetRatings(cur, setFName)
  
  (users, items) = getUserItems(setFName)
  print 'nUsers: ', len(users), ' nItems: ', len(items)

  writeList(list(users), opPrefix + '_set_users.txt')
  writeList(list(items), opPrefix + '_set_items.txt')

  uiRatFName = opPrefix + '_ui_ratings.csv'
  uiRat = getUserItemRatings(cur, users, items, uiRatFName)

  invalUserByTS = getInvalidUsersByTS(setFName)
  invalUserByRat = getInvalidUsersWRat(uiRat, uSetRatings)
  writeList(list(invalUserByRat), opPrefix + '_invalU_by_rat.txt')

  invalUsers = invalUserByTS.union(invalUserByRat)
  print "Total invalid users: ", len(invalUsers) 
  writeList(list(invalUsers), opPrefix + '_invalid_users.txt')

  filtSetRatings(uSetRatings, invalUsers, opPrefix + "_filt_set_ratings.csv")


if __name__ == '__main__':
  main()



