import random
import os
import numpy as np

NUSERS      = 854
NITEMS      = 12549
#FACDIMS     = [1, 5, 10, 15, 25, 50, 75, 100]
FACDIMS     = [1, 10, 15, 25, 50, 75, 100]
REGS        = [0.001, 0.01, 0.1, 1, 10] 

uREGS       = REGS #[0.1, 1]
iREGS       = REGS #[0.001, 0.01]
ubiasREGS   = REGS #[0.001, 0.01, 1, 10]
ibiasREGS   = REGS #[0.01, 10]
#biasREGS   = [0.001, 0.01]
setBiasRegs = REGS #[0.001, 0.01, 1, 10]

#TODO add 1 as it goes till 0.75
GAMMAS      = np.arange(-1, 1.25, 0.25)
GAMMAS      = list(GAMMAS) + [-0.05, 0.05]
LEARNRATE   = 0.001
SEED        = 1

DATA         = "/home/karypisg/msharma/data/setrec/movielens/split5/"

SETREC       = "/home/karypisg/msharma/dev/setrec/cppsrc/setrecRankBPRTop_30"
PREFIX       = "rankBPRTop_30"
OPDIR        = DATA + PREFIX 

TRAIN_SET    = DATA + "ml_set.train.lfs"
TEST_SET     = DATA + "ml_set.test.lfs"
VAL_SET      = DATA + "ml_set.val.lfs"

RATMAT       = DATA + "ml_ratings.csr"
TRAIN_RATMAT = DATA + "train.csr"
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
      usetbiasreg = REGS[random.randint(0, len(REGS)-1)]
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
    for ureg in uREGS:
      for ireg in iREGS:
        for ubiasreg in ubiasREGS:
          for ibiasreg in ibiasREGS:
            for usetbiasreg in setBiasRegs:
            #for usetbiasreg in [0]:
                jobStr      = '_'.join(map(str, [ureg, ireg, ubiasreg, ibiasreg, 
                  usetbiasreg, dim]))
                print SETREC, NUSERS, NITEMS, dim, 5000, SEED, \
                    ureg, ireg, usetbiasreg, ubiasreg, ibiasreg, 0, LEARNRATE, 0, 0, \
                    TRAIN_SET, TEST_SET, VAL_SET, RATMAT, TRAIN_RATMAT, TEST_RATMAT, \
                    VAL_RATMAT, PREFIX, \
                    " > " + OPDIR + "/" + PREFIX + "_" + jobStr + ".txt"


def genRankRegJobs():
  for dim in FACDIMS:
    for ureg in uREGS:
      for ireg in iREGS:
        for ibiasreg in ibiasREGS:
          gamma = 0
          #for gamma in GAMMAS:
          #for gamma in [0]:
          jobStr      = '_'.join(map(str, [ureg, ireg, 0, ibiasreg, 
            0, dim]))
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


def genJobs():
  params = [
      [10,0.001,0.001,0.01,0,75],
      [0.1,0.01,0.001,10,0,1],
      [10,0.01,0.001,0.01,0,75],
      [10,0.001,0.001,0.01,0,100],
      [0.1,0.01,0.001,1,0,1],
      [10,0.01,0.01,0.01,0,75],
      [10,0.001,0.01,0.01,0,75],
      [1,0.01,0.001,0.01,0,15],
      [0.1,0.01,0.01,10,0,1],
      [1,0.01,0.001,0.01,0,75],
      [10,0.001,0.01,0.01,0,100],
      [1,0.01,0.001,0.01,0,100],
      [10,0.01,0.001,0.01,0,100],
      [1,0.001,0.001,0.01,0,15],
      [1,0.001,0.001,0.01,0,100],
      [10,0.001,0.001,0.01,0,15],
      [1,0.001,0.001,0.01,0,75],
      [1,0.001,0.001,0.01,0,50],
      [1,0.01,0.001,0.01,0,50],
      [0.1,0.001,0.001,1,0,1]
      ]
  for param in params[:5]:
    ureg  = param[0]
    ireg  = param[1]
    ubiasreg = param[2]
    ibiasreg = param[3]
    usetbiasreg = param[4]
    dim   = param[5]
    for trainMat in ['train_0.1.csr', 'train_0.25.csr', 'train_0.5.csr',
        'train_0.75.csr', 'train.csr']:
      jobStr      = '_'.join(map(str, [ureg, ireg, ubiasreg, ibiasreg, 
        0, dim, trainMat]))
      trainMat = DATA + trainMat
      print SETREC, NUSERS, NITEMS, dim, 5000, SEED, \
          ureg, ireg, usetbiasreg, ubiasreg, ibiasreg, 0, LEARNRATE, 0, 0, \
          TRAIN_SET, TEST_SET, VAL_SET, RATMAT, trainMat, TEST_RATMAT, \
          VAL_RATMAT, PREFIX, \
          " > " + OPDIR + "/" + PREFIX + "_" + jobStr + ".txt"
    

genRankRegJobs() 


