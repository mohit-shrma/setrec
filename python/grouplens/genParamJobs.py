import sys
import os


DATA      = "/home/karypisg/msharma/data/setrec/movielens"
NUSERS    = 854
NITEMS    = 12549
LEARNRATE = 0.001
SEED      = 1

def genJobs(paramsFile, prefix, setrec):
  with open(paramsFile, 'r') as f:
    for line in f:
      cols        = line.strip().split()
      
      if len(cols) < 6 or len(cols) > 6:
        print 'Err: ncols:', len(cols), cols
        return 
      uReg        = float(cols[0])
      iReg        = float(cols[1])
      uBiasReg    = float(cols[2])
      iBiasReg    = float(cols[3])
      sessBiasReg = float(cols[4])
      dim         = int(cols[5])
      
      jobStr      = '_'.join(map(str, [uReg, iReg, uBiasReg, iBiasReg,
        sessBiasReg, dim]))
      
      for splitInd in range(1,6):
        trainSet = os.path.join(DATA, "split" + str(splitInd),
            "ml_set.train.lfs")
        testSet = os.path.join(DATA, "split" + str(splitInd), 
            "ml_set.test.lfs")
        valSet = os.path.join(DATA, "split" + str(splitInd), 
            "ml_set.val.lfs")
        ratMat = os.path.join(DATA, "split" + str(splitInd), 
            "ml_ratings.csr")
        trainMat = os.path.join(DATA, "split" + str(splitInd), 
            "train.csr")
        testMat = os.path.join(DATA, "split" + str(splitInd), 
            "test.csr")
        valMat = os.path.join(DATA, "split" + str(splitInd), 
            "val.csr")
        
        opDir = os.path.join(DATA, "split" + str(splitInd), 
            prefix)

        if not os.path.exists(opDir):
          os.mkdir(opDir)
        print setrec, NUSERS, NITEMS, dim, 5000, SEED, \
            uReg, iReg, sessBiasReg, uBiasReg, iBiasReg, 0, LEARNRATE, 0, 0, \
            trainSet, testSet, valSet, ratMat, trainMat, testMat, valMat, prefix, \
            " > " + opDir + "/" + prefix + "_" + jobStr + ".txt"


def main():
  paramsFile = sys.argv[1]
  prefix     = sys.argv[2]
  setrec     = sys.argv[3]
  genJobs(paramsFile, prefix, setrec)

if __name__ == '__main__':
  main()


