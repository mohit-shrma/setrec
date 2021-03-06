import os

#ureg, ireg, ubiasreg, ibiasreg, usessbiasreg, gamma, dim
MF_PARAMS           = [0.1, 0.01, 0.001, 1, 0, 0, 25]
LFS_PARAMS          = [1, 0.001, 0.001, 0.01, 0, 0, 75]
LFS_GBIAS_PARAMS    = [1, 0.001, 0.1, 0.01, 10, 0, 100]
LFS_SESSBIAS_PARAMS = [1, 0.001, 0.001, 0.01, 0.01, 0, 100]
LFS_BPR             = [1, 0.01, 0, 0.001, 0, 0, 75]



PARENT_DATA = "/home/grad02/mohit/exmoh/movielens/" 

SETREC      = "/home/grad02/mohit/exmoh/setrec/cppsrc/setrecAvg"
PARAMS      = LFS_PARAMS
PREFIX      = "avg"

NUSERS      = 854
NITEMS      = 12549
SEED        = 1
LEARNRATE   = 0.001


def genJob(setrec, ureg, ireg, ubiasreg, ibiasreg, usetbiasreg, gamma, dim, ratMat, 
    trainMat, testMat, valMat, trainSet, testSet, valSet, opdir, suffix):
  jobStr      = '_'.join(map(str, [ureg, ireg, ubiasreg, ibiasreg, 
    usetbiasreg, gamma, dim]))
  print setrec, NUSERS, NITEMS, dim, 5000, SEED, ureg, ireg, usetbiasreg, \
      ubiasreg, ibiasreg, 0, LEARNRATE, 0, gamma, trainSet, testSet, valSet, \
      ratMat, trainMat, testMat, valMat, PREFIX, " > " + opdir + "/" + PREFIX + \
      "_" + jobStr + "_" + suffix + ".txt"


for splitInd in range(1, 6):
  DATA = os.path.join(PARENT_DATA, "split" + str(splitInd))
  OPDIR = os.path.join(DATA, PREFIX)

  if not os.path.exists(OPDIR):
    os.mkdir(OPDIR)

  TRAIN_SET    = os.path.join(DATA, "ml_set.train.lfs")
  TEST_SET     = os.path.join(DATA, "ml_set.test.lfs")
  VAL_SET      = os.path.join(DATA, "ml_set.val.lfs")
  RATMAT       = os.path.join(DATA, "ml_ratings.csr")
  TRAIN_RATMAT = os.path.join(DATA, "train.csr")
  suffix      = "1.0"
  TEST_RATMAT  = os.path.join(DATA, "test.csr")
  VAL_RATMAT   = os.path.join(DATA, "val.csr")

  ureg        = PARAMS[0]
  ireg        = PARAMS[1]
  ubiasreg    = PARAMS[2]
  ibiasreg    = PARAMS[3]
  usetbiasreg = PARAMS[4]
  gamma       = PARAMS[5]
  dim         = PARAMS[6]

  genJob(SETREC, ureg, ireg, ubiasreg, ibiasreg, usetbiasreg, gamma, dim,
      RATMAT, TRAIN_RATMAT, TEST_RATMAT, VAL_RATMAT, TRAIN_SET, TEST_SET,
      VAL_SET, OPDIR, suffix)





