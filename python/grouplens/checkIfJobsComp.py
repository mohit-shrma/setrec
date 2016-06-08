import sys
import os


def checkIfFinished(ipFName):
  with open(ipFName, 'r') as f:
    for line in f:
      if line.startswith('Val NDCG'):
        return True
  return False


def dispIncomp(fileList):
  with open(fileList, 'r') as f:
    for line in f:
      fName = line.strip()
      if os.path.isfile(fName) and checkIfFinished(fName):
        print 'Done: ', fName
      else:
        print 'NotDone: ', fName


def main():
  ipFName = sys.argv[1]
  dispIncomp(ipFName)


if __name__ == '__main__':
  main()


