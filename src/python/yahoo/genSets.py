import sys


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
  g.write(str(user) + ' ' + str(nTracks) + ' ' 
      + ' '.join(map(str, userTracks)) + '\n')
  for album, ratTracks in userAlbumRatTracks.iteritems():
    albumTracks = ratTracks[0]
    albumRat = ratTracks[1]
    g.write(str(albumRat) + ' ' + str(len(albumTracks)) + ' ' 
        + ' '.join(map(str, albumTracks)) + '\n')


def writeSets(fName, opPrefix):
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
      userTracks.append(track)
      if album not in userAlbumRatTracks:
        userAlbumRatTracks[album] = [[], 0]
      userAlbumRatTracks[album][0].append(track)
      userAlbumRatTracks[album][1] = albumRat
    #write last user
    writeUserSet(currUser, userTracks, userAlbumRatTracks, g)


def writeRatings(fName, opPrefix):
  trackRatOpName = opPrefix + '_trackRat.txt'
  albumRatOpName = opPrefix + '_albumRat.txt'
  with open(fName, 'r') as f, open(trackRatOpName, 'w') as g, open(albumRatOpName, 'w') as h:
    for line in f:
      cols     = map(int, line.strip().split())
      user     = cols[0]
      album    = cols[1]
      albumRat = cols[2]
      track    = cols[3]
      trackRat = cols[4]
      g.write(str(user) + ' ' + str(track) + ' ' + str(trackRat) + '\n')  
      h.write(str(user) + ' ' + str(album) + ' ' + str(albumRat) + '\n')
      

def main():
  
  yRatFName  = sys.argv[1]
  opPrefix   = sys.argv[2]

  #get user and track map
  (uMap, trackMap, albumMap) = getUserTrackAlbumMap(yRatFName)
  writeMap(uMap, opPrefix + '_userMap.txt')
  writeMap(trackMap, opPrefix + '_trackMap.txt')
  writeMap(albumMap, opPrefix + '_albumMap.txt')

  #convert passed rating file 
  convFName = convRatFName(yRatFName, uMap, trackMap, albumMap, opPrefix)
  
  #use converted file to generate sets
  #NOTE: this assumes that file is in sorted order
  writeSets(convFName, opPrefix)  
  
  #write track and album rating triplets
  writeRatings(convFName, opPrefix)


if __name__ == '__main__':
  main()



