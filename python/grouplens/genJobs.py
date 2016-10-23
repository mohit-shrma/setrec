import sys
import random
import os

NUSERS      = 854
NITEMS      = 12549
FACDIMS     = [1, 5, 10, 25, 50, 75, 100]
#FACDIMS     = [1, 10, 15, 25, 50, 75, 100]
REGS        = [0.001, 0.01, 0.1, 1, 10] 

uREGS       = REGS #[0.1, 1]
iREGS       = REGS #[0.001, 0.01]
ubiasREGS   = REGS #[0.001, 0.01, 1, 10]
ibiasREGS   = REGS #[0.01, 10]
gbiasREGS   = [0]#REGS
#biasREGS   = [0.001, 0.01]
setBiasRegs = REGS #[0.001, 0.01, 1, 10]

#TODO add 1 as it goes till 0.75
#GAMMAS      = np.arange(-1, 1.25, 0.25)
#GAMMAS      = list(GAMMAS) + [-0.05, 0.05]
LEARNRATE   = 0.001
SEED        = 1

DATA         = "/home/karypisg/msharma/data/setrec/movielens/fiveSplits/split1/"

SETREC       = "/home/karypisg/msharma/dev/setrec/cppsrc/setrecFMEntropy"
PREFIX       = "setFMEntropy"
OPDIR        = DATA + PREFIX 

TRAIN_SET    = "ml_set.train.lfs"
TEST_SET     = "ml_set.test.lfs"
VAL_SET      = "ml_set.val.lfs"

RATMAT       = "ml_ratings.csr"
TRAIN_RATMAT = "train.csr"
TEST_RATMAT  = "test.csr"
VAL_RATMAT   = "val.csr"

TRAIN_MIX_MATS = ['train_0.010000.csr', 'train_0.250000.csr', 'train_0.500000.csr', 'train_0.750000.csr', TRAIN_RATMAT]

def genGridRegJobs(setrec, prefix, data):
  opDir = os.path.join(data, prefix)
  if not os.path.exists(opDir):
    os.mkdir(opDir)
  for dim in FACDIMS:
    for ureg in REGS:
      for ireg in REGS:
        for usetbiasreg in [0]:#REGS:#setBiasRegs:
          for gamma in [0]:#[0.1, 0.25, 0.5]:
            jobStr = '_'.join(map(str, [dim, ureg, ireg, usetbiasreg, gamma]))
            print 'cd ' + data + ' && ', setrec, NUSERS, NITEMS, dim, 5000, SEED, \
                ureg, ireg, usetbiasreg, 0, 0, 0, \
                LEARNRATE, 0, gamma, TRAIN_SET, TEST_SET, VAL_SET, RATMAT, \
                TRAIN_RATMAT, TEST_RATMAT, VAL_RATMAT, prefix, \
                " > " + opDir + "/" + prefix + "_" + jobStr + ".txt"


def genGridMixRegJobs(setrec, prefix, data):
  isMixRat = 1
  
  opDir = os.path.join(data, prefix)
  if not os.path.exists(opDir):
    os.mkdir(opDir)
  
  for trainMatInd in range(len(TRAIN_MIX_MATS)):
    trainRatMat = TRAIN_MIX_MATS[trainMatInd]
    for dim in FACDIMS:
      for ureg in REGS:
        for ireg in REGS:
          for usetbiasreg in [0]:#setBiasRegs:#for var, maxmin
            for gamma in [0]:#[0.25, 0.5]:#for var
              jobStr = '_'.join(map(str, [dim, ureg, ireg, usetbiasreg, gamma]))
              print 'cd ' + data + ' && ', setrec, NUSERS, NITEMS, 5, 5000, SEED, \
                  ureg, ireg, usetbiasreg, 0, 0, 0, \
                  LEARNRATE, isMixRat, gamma, TRAIN_SET, TEST_SET, VAL_SET, RATMAT, \
                  trainRatMat, TEST_RATMAT, VAL_RATMAT, prefix, \
                  " > " + opDir + "/" + prefix + "_" + str(trainMatInd)  + \
                  "_mix_" + jobStr + ".txt"



def main():
  setrec = sys.argv[1]
  data = sys.argv[2]
  prefix = os.path.basename(setrec)

  for i in range(1,2):
    subDirName = 'split' + str(i)
    subDir = os.path.join(data, subDirName)
    if not os.path.exists(subDir):
      print 'Path not found: ', subDir
    else:
      genGridRegJobs(setrec, prefix, subDir)

if __name__ == '__main__':
  main()
