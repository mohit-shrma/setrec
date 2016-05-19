import sys
import random


def writeSampCSR(ipFName, opFName, nRatPerUser):
  with open(ipFName, 'r') as f, open(opFName, 'w') as g:
    for line in f:
      cols = line.strip().split()
      itemRatings = []
      for i in range(0, len(cols), 2):
        itemRatings.append((cols[i], cols[i+1]))
      random.shuffle(itemRatings)
      for (item, rating) in itemRatings[:nRatPerUser]:
        g.write(item + ' ' + rating + ' ')
      g.write('\n')


def main():
  ipFName     = sys.argv[1]
  nRatPerUser = int(sys.argv[2])
  opFName     = sys.argv[3]
  writeSampCSR(ipFName, opFName, nRatPerUser)


if __name__ == '__main__':
  main()

