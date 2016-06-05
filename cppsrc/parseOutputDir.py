import sys
import os
from os import listdir
from os.path import isfile, join

def parseFiles(dirName, prefix):
  ipFiles = [f for f in listdir(dirName) if isfile(join(mypath, f))]
  for ipFName in ipFiles:
    with open(ipFName, 'r') as f:
      for line in f:
        if line.startswith(prefix):
          cols = line.strip().split()
          val = float(cols[-1])
          fName = os.path.basename(ipFName)
          params = fname.split('_')
          print params[1], val
          continue

def main():
  dirName = sys.argv[1]
  
  prefix = ''
  for s in sys.argv[2:]:
    prefix = prefix + s + ' '
  
  



if __name__ == '__main__':
  main()


