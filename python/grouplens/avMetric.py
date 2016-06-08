import sys


def avgMetric(ipFNames, metric):
  avg = 0.0
  found = 0.0
  for fName in ipFNames:
    with open(fName, 'r') as f:
      for line in f:
        if line.startswith(metric):
          cols = line.strip().split()
          avg += float(cols[-1])
          found += 1
  print found, avg/found



def main():
  ipFNames = sys.argv[1:-1]
  metric = sys.argv[-1]
  print ipFNames, metric
  avgMetric(ipFNames, metric)

if __name__ == '__main__':
  main()

