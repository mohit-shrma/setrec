import sys


def getItemInSets(iMapFName):
  setItems = set([])
  with open(iMapFName, 'r') as f:
    cols = line.strip().split(',')
    setItems.add(int(cols[0]))
  return setItems


def getUItemCount(ipRatFName, setItems):
  f = open(ipRatFName, 'r')
  head = f.readline()
  for line in f:
    cols = line.strip().split(',')
    
  f.close()

def main():
  ipRatFName = sys.argv[1]
  iMapFName  = sys.argv[2]
  uMapFName  = sys.argv[3]


if __name__ == '__main__':
  main()


