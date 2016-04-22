import sys
import random

def convRatFName(yRatFName, uMap, trackMap, albumMap, opPrefix):
  opFName = opPrefix + '_ratConv.txt'
  with open(yRatFName, 'r') as f, open(opFName, 'w') as g:
    for line in f:
      cols     = map(int, line.strip().split(','))
      user     = cols[0]
      album    = cols[1]
      albumRat = cols[2]
      track    = cols[3]
      trackRat = cols[4]
      g.write(str(uMap[user]) + ' ' + str(albumMap[album]) + ' ' + str(albumRat)
          + ' ' + str(trackMap[track]) + ' ' + str(trackRat) + '\n') 
  return opFName


def getUserTrackAlbumMap(yRatFName):  
  uMap     = {}
  trackMap = {}
  albumMap = {}
  u        = 0
  t        = 0
  a        = 0 
  with open(yRatFName, 'r') as f:
    for line in f:
      cols     = map(int, line.strip().split(','))
      user     = cols[0]
      album    = cols[1]
      albumRat = cols[2]
      track    = cols[3]
      trackRat = cols[4]
      if user not in uMap:
        uMap[user] = u
        u += 1
      if track not in trackMap:
        trackMap[track] = t
        t += 1
      if album not in albumMap:
        albumMap[album] = a
        a += 1
  return (uMap, trackMap, albumMap)


def writeMap(m, fName):
  with open(fName, 'w') as g:
    for k, v in m.iteritems():
      g.write(str(k) + ' ' + str(v) + '\n')


def writeUserSet(user, userTracks, userAlbumRatTracks, g):
  nTracks = len(userTracks)
  g.write(str(user) + ' ' + str(len(userAlbumRatTracks)) + ' ' + str(nTracks) 
      + ' ' + ' '.join(map(str, userTracks)) + '\n')
  for album, ratTracks in userAlbumRatTracks.iteritems():
    albumTracks = ratTracks[0]
    albumRat = ratTracks[1]
    g.write(str(albumRat) + ' ' + str(len(albumTracks)) + ' ' 
        + ' '.join(map(str, albumTracks)) + '\n')


def writeSets(fName, userTestTracks, userValTracks, opPrefix):
  opName = opPrefix + '_set.txt'
  with open(fName, 'r') as f, open(opName, 'w') as g:
    userTracks = []
    userAlbumRatTracks = {}
    currUser = None
    for line in f:
      cols     = map(int, line.strip().split(' '))
      user     = cols[0]
      album    = cols[1]
      albumRat = cols[2]
      track    = cols[3]
      trackRat = cols[4]
      if currUser is None:
        currUser = user
      if user != currUser:
        #write out previous user info 
        writeUserSet(currUser, userTracks, userAlbumRatTracks, g)
        #reset
        userTracks = []
        userAlbumRatTracks = {}
        currUser = user
      if track in userTestTracks[user] or track in userValTracks[user]:
        continue
      userTracks.append(track)
      if album not in userAlbumRatTracks:
        userAlbumRatTracks[album] = [[], 0]
      userAlbumRatTracks[album][0].append(track)
      userAlbumRatTracks[album][1] = albumRat
    #write last user
    writeUserSet(currUser, userTracks, userAlbumRatTracks, g)


def writeRatings(fName, opPrefix):
  userTestTracks = {}
  userValTracks  = {}
  trackRatOpName = opPrefix + '_trackRat.txt'
  albumRatOpName = opPrefix + '_albumRat.txt'
  userAlbums     = {}
  with open(fName, 'r') as f, open(trackRatOpName, 'w') as g, open(albumRatOpName, 'w') as h:
    for line in f:
      cols     = map(int, line.strip().split())
      user     = cols[0]
      album    = cols[1]
      albumRat = cols[2]
      track    = cols[3]
      trackRat = cols[4]
      g.write(str(user) + ' ' + str(track) + ' ' + str(trackRat) + '\n')  
      
      if user not in userTestTracks:
        if random.randint(0,10) % 2 == 0:
          userTestTracks[user] = [track, trackRat]
      elif user not in userValTracks:
        if random.randint(0,10) % 2 == 0:
          userValTracks[user] = [track, trackRat]
      
      if user not in userAlbums:
        userAlbums[user] = set([])
      if album not in userAlbums[user]:
        userAlbums[user].add(album)
        h.write(str(user) + ' ' + str(album) + ' ' + str(albumRat) + '\n')
  return (userTestTracks, userValTracks)


def writeTrainTestValTriplets(fName, opPrefix, userTestTracks, userValTracks):
  trainFName = opPrefix + '_train.triplet'
  testFName  = opPrefix + '_test.triplet'
  valFName   = opPrefix + '_val.triplet'
  with open(fName, 'r') as f, open(trainFName, 'w') as g,\
      open(testFName, 'w') as h, open(valFName, 'w') as p:
        for line in f:
          cols     = map(int, line.strip().split())
          user     = cols[0]
          album    = cols[1]
          albumRat = cols[2]
          track    = cols[3]
          trackRat = cols[4]
          if track == userTestTracks[user][0]:
            h.write(str(user) + ' ' + str(track) + ' ' + str(trackRat) + '\n')
          elif track == userValTracks[user][0]:
            p.write(str(user) + ' ' + str(track) + ' ' + str(trackRat) + '\n')
          else:
            g.write(str(user) + ' ' + str(track) + ' ' + str(trackRat) + '\n')



def main():
  yRatFName = sys.argv[1]
  opPrefix  = sys.argv[2]
  seed      = int(sys.argv[3])

  opPrefix = opPrefix + '_' + str(seed) 

  random.seed(seed)

  #get user and track map
  (uMap, trackMap, albumMap) = getUserTrackAlbumMap(yRatFName)
  writeMap(uMap, opPrefix + '_userMap.txt')
  writeMap(trackMap, opPrefix + '_trackMap.txt')
  writeMap(albumMap, opPrefix + '_albumMap.txt')

  print 'Written maps...'

  #convert passed rating file 
  convFName = convRatFName(yRatFName, uMap, trackMap, albumMap, opPrefix)
  
  #convFName = opPrefix + '_ratConv.txt' 
  print 'Converted passed rating file...'

  #write track and album rating triplets
  (userTestTracks, userValTracks) = writeRatings(convFName, opPrefix)
 
  print 'Wrote album and track ratings'

  #make sure user testTRacks and val tracks are not empty
  users = uMap.values()
  for u in users:
    if u not in userTestTracks:
      print 'User not found in test: ', u
    if u not in userValTracks:
      print 'User not found in val: ', u

  #write out training, test and validation triplets
  writeTrainTestValTriplets(convFName, opPrefix, userTestTracks, userValTracks) 
  
  print 'Train, test and val written...'

  #use converted file to generate sets
  #NOTE: this assumes that file is in sorted order
  writeSets(convFName, userTestTracks, userValTracks, opPrefix)  
  

if __name__ == '__main__':
  main()



