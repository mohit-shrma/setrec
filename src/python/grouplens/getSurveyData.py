import sys
import MySQLdb

def writeList(lst, opFName):
  with (opFName, 'w') as g:
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


def getSetRatings(cur, opFileName):
  #query to get all the data
  cur.execute('select * from expt_set_user_rating')
  with open(opFileName, 'w') as g:
    for row in cur.fetchall():
      strRow = map(str, row)
      g.write(','.join(strRow) + '\n')


def getUserItems(uSetCSVFName):
  users = set([])
  items = set([])
  with open(uSetCSVFName, 'r') as f:
    for line in f:
      cols = line.strip().split(',')
      user = int(cols[1])
      uItems = map(int, cols[2].split('-'))
      users.add(user)
      for uItem in uItems:
        items.add(uItem)
  return (users, items)


def getUserItemRatings(cur, users, items, opFileName):
  #query to get join of ratings and set_ratings
  cur.execute('select A.userId, A.movieId, A.rating, A.tstamp from \
      user_rating_pairs A INNER JOIN expt_set_user_rating B ON \
      A.userId=B.userId;')
  with open(opFileName, 'w') as g:
    for row in cur.fetchall():
      if row[0] in users and row[1] in items:
        strRow = map(str, row)
        g.write(','.join(strRow) + '\n')


def main():
  opPrefix = sys.argv[1]
  cur = getDBCursor()
  
  setFName = opPrefix + '_set_ratings.csv'
  getSetRatings(cur, setFName)
  
  (users, items) = getUserItems(setFName)
  
  print 'nUsers: ', len(users), ' nItems: ', len(items)

  writeList(list(users), opPrefix + '_set_users.txt')
  writeList(list(items), opPrefix + '_set_items.txt')

  uiRatFName = opPrefix + '_ui_ratings.csv'
  getUserItemRatings(cur, users, items, uiRatFName)


if __name__ == '__main__':
  main()


