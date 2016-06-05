import sys
import os
from os import listdir
from os.path import isfile, join

def parseFiles(dirName, prefix):
  ipFiles = [f for f in listdir(dirName) if isfile(join(dirName, f))]
  for ipFName in ipFiles:
    with open(join(dirName, ipFName), 'r') as f:
      for line in f:
        if line.startswith(prefix):
          cols = line.strip().split()
          val = float(cols[-1])
          fName = os.path.basename(ipFName)
          params = fName.split('_')
          print '\t'.join(params), val
          continue

def main():
  dirName = sys.argv[1]
  
  prefix = ''
  for s in sys.argv[2:]:
    prefix = prefix + s + ' '
  prefix = prefix.strip()
  #print 'prefix:', prefix 
  parseFiles(dirName, prefix)  



if __name__ == '__main__':
  main()


