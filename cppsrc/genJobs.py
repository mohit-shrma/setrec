import random
import os
import numpy as np

NUSERS      = 854
NITEMS      = 12549
FACDIMS     = [1, 5, 10, 15, 25, 50, 75, 100]
#FACDIMS     = [1, 10, 15, 25, 50, 75, 100]
REGS        = [0.001, 0.01, 0.1, 1, 10] 

uiREGS      = [0.001, 0.01, 0.1, 1, 10]
biasREGS    = [0.001]
#biasREGS   = [0.001, 0.01]
setBiasRegs = [0.001, 0.01, 0.1]

#TODO add 1 as it goes till 0.75
GAMMAS      = np.arange(-1, 1.25, 0.25)
GAMMAS      = list(GAMMAS) + [-0.05, 0.05]
LEARNRATE   = 0.001
SEED        = 1

DATA         = "/home/karypisg/msharma/data/setrec/movielens/split_50p/"

SETREC       = "/home/karypisg/msharma/dev/setrec/cppsrc/setrecMF"
PREFIX       = "setrecMF"
OPDIR        = DATA + "indivmf" 

TRAIN_SET    = DATA + "ml_set.train.lfs"
TEST_SET     = DATA + "ml_set.test.lfs"
VAL_SET      = DATA + "ml_set.val.lfs"

RATMAT       = DATA + "ml_ratings.csr"
TRAIN_RATMAT = DATA + "train_50.csr"
TEST_RATMAT  = DATA + "test.csr"
VAL_RATMAT   = DATA + "val.csr"

if not os.path.exists(OPDIR):
  os.mkdir(OPDIR)


def genRandRegJobs():
  for dim in FACDIMS:
    nJobs       = 0
    while nJobs < 100:
      ureg        = REGS[random.randint(0, len(REGS)-1)]
      ireg        = REGS[random.randint(0, len(REGS)-1)]
      ubiasreg    = REGS[random.randint(0, len(REGS)-1)]
      ibiasreg    = REGS[random.randint(0, len(REGS)-1)]
      usetbiasreg = 0 #REGS[random.randint(0, len(REGS)-1)]
      jobStr      = '_'.join(map(str, [ureg, ireg, ubiasreg, ibiasreg,
        usetbiasreg, dim]))
      print SETREC, NUSERS, NITEMS, dim, 5000, SEED, \
          ureg, ireg, usetbiasreg, ubiasreg, ibiasreg, 0, LEARNRATE, 0, 0, \
          TRAIN_SET, TEST_SET, VAL_SET, RATMAT, TRAIN_RATMAT, TEST_RATMAT, \
          VAL_RATMAT, PREFIX, \
          " > " + OPDIR + "/" + PREFIX + "_" + jobStr + ".txt"

      nJobs += 1
    

def genGridRegJobs():
  for dim in FACDIMS:
    for ureg in uiREGS:
      for ireg in uiREGS:
        for ubiasreg in biasREGS:
          for ibiasreg in biasREGS:
            for usetbiasreg in biasREGS:
              jobStr      = '_'.join(map(str, [ureg, ireg, ubiasreg, ibiasreg, 
                usetbiasreg, dim]))
              print SETREC, NUSERS, NITEMS, dim, 5000, SEED, \
                  ureg, ireg, usetbiasreg, ubiasreg, ibiasreg, 0, LEARNRATE, 0, 0, \
                  TRAIN_SET, TEST_SET, VAL_SET, RATMAT, TRAIN_RATMAT, TEST_RATMAT, \
                  VAL_RATMAT, PREFIX, \
                  " > " + OPDIR + "/" + PREFIX + "_" + jobStr + ".txt"


def genRankRegJobs():
  for dim in FACDIMS:
    for ureg in uiREGS:
      for ireg in uiREGS:
        for ibiasreg in biasREGS:
          for gamma in GAMMAS:
            jobStr      = '_'.join(map(str, [ureg, ireg, 0, ibiasreg, 
              0, gamma, dim]))
            print SETREC, NUSERS, NITEMS, dim, 5000, SEED, \
                ureg, ireg, 0, 0, ibiasreg, 0, LEARNRATE, 0, gamma, \
                TRAIN_SET, TEST_SET, VAL_SET, RATMAT, TRAIN_RATMAT, TEST_RATMAT, \
                VAL_RATMAT, PREFIX, \
                " > " + OPDIR + "/" + PREFIX + "_" + jobStr + ".txt"


def genRandRankRegJobs():
  for dim in FACDIMS:
    nJobs       = 0
    while nJobs < 100:
      ureg        = REGS[random.randint(0, len(REGS)-1)]
      ireg        = REGS[random.randint(0, len(REGS)-1)]
      ibiasreg    = REGS[random.randint(0, len(REGS)-1)]
      gamma       = GAMMAS[random.randint(0, len(GAMMAS)-1)]
      jobStr      = '_'.join(map(str, [ureg, ireg, 0, ibiasreg,
        0, gamma, dim]))
      print SETREC, NUSERS, NITEMS, dim, 5000, SEED, \
          ureg, ireg, 0, 0, ibiasreg, 0, LEARNRATE, 0, gamma, \
          TRAIN_SET, TEST_SET, VAL_SET, RATMAT, TRAIN_RATMAT, TEST_RATMAT, \
          VAL_RATMAT, PREFIX, \
          " > " + OPDIR + "/" + PREFIX + "_" + jobStr + ".txt"

      nJobs += 1



genRandRegJobs() 

