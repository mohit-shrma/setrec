import os
import sys

REGS        = [0.001, 0.01, 0.1, 1, 10]
NUSERS      = 6040
NITEMS      = 3706
SEED        = 1
LEARNRATE   = 0.001
RATMAT      = "ratings.syn.csr"
TEST_RATMAT = "test.syn.csr"
VAL_RATMAT  = "val.syn.csr"
TRAIN_RATMATS = ["train_0.syn.csr", "train_1.syn.csr", "train_2.syn.csr", \
    "train_3.syn.csr", "train_4.syn.csr"]


def genGridRegJobs(setrec, prefix, data, nSplits=5):
  for i in range(nSplits):
    trainSet    = "split" + str(i) + "/" + "ml_set.train.syn_" + str(i) + ".lfs"
    testSet     = "split" + str(i) + "/" + "ml_set.test.syn_" + str(i) + ".lfs"
    valSet      = "split" + str(i) + "/" + "ml_set.val.syn_" + str(i) + ".lfs"
    trainRatMat = "train.syn.csr"
    opDir = os.path.join(data, "split" + str(i), prefix)
    if not os.path.exists(opDir):
      os.mkdir(opDir)

    for ureg in REGS:
      for ireg in REGS:
        for usetbiasreg in [0]:#REGS:#setBiasRegs:
          for gamma in [0]:#[0.25, 0.5]:
            jobStr = '_'.join(map(str, [ureg, ireg, usetbiasreg, gamma]))
            print 'cd ' + data + ' && ', setrec, NUSERS, NITEMS, 5, 5000, SEED, \
                ureg, ireg, usetbiasreg, 0, 0, 0, \
                LEARNRATE, 0, gamma, trainSet, testSet, valSet, RATMAT, \
                trainRatMat, TEST_RATMAT, VAL_RATMAT, prefix, \
                " > " + opDir + "/" + prefix + "_" + jobStr + ".txt"


def genMixJobsFromParam(setrec, prefix, data, topParamFName, nSplits=5):
  isMixRat = 1
  with open(topParamFName, 'r') as f:
    for line in f:
      cols = line.strip().split()
    for i in range(nSplits):
      trainSet    = "split" + str(i) + "/" + "ml_set.train.syn_" + str(i) + ".lfs"
      testSet     = "split" + str(i) + "/" + "ml_set.test.syn_" + str(i) + ".lfs"
      valSet      = "split" + str(i) + "/" + "ml_set.val.syn_" + str(i) + ".lfs"
      opDir = os.path.join(data, "split" + str(i), prefix)
      if not os.path.exists(opDir):
        os.mkdir(opDir)
      for trainMatInd in range(5):
        trainRatMat = "train_" + str(trainMatInd) + ".syn.csr"
        ureg = float(cols[0])
        ireg = float(cols[1])
        usetbiasreg = float(cols[2])
        gamma = float(cols[3])
        jobStr = '_'.join(map(str, [ureg, ireg, usetbiasreg, gamma]))
        print 'cd ' + data + ' && ', setrec, NUSERS, NITEMS, 5, 5000, SEED, \
            ureg, ireg, usetbiasreg, 0, 0, 0, \
            LEARNRATE, isMixRat, gamma, trainSet, testSet, valSet, RATMAT, \
            trainRatMat, TEST_RATMAT, VAL_RATMAT, prefix, \
            " > " + opDir + "/" + prefix + "_" + str(trainMatInd)  + \
            "_mix_" + jobStr + ".txt"



def genGridMixRegJobs(setrec, prefix, data, nSplits=5):
  isMixRat = 1
  for i in range(nSplits):
    trainSet    = "split" + str(i) + "/" + "ml_set.train.syn_" + str(i) + ".lfs"
    testSet     = "split" + str(i) + "/" + "ml_set.test.syn_" + str(i) + ".lfs"
    valSet      = "split" + str(i) + "/" + "ml_set.val.syn_" + str(i) + ".lfs"
    opDir = os.path.join(data, "split" + str(i), prefix)
    if not os.path.exists(opDir):
      os.mkdir(opDir)
    for trainMatInd in range(5):
      trainRatMat = "train_" + str(trainMatInd) + ".syn.csr"
      for ureg in REGS:
        for ireg in REGS:
          for usetbiasreg in [0]:#REGS:#setBiasRegs:#for var, maxmin
            for gamma in [0]:#[0.25, 0.5]:#for var
              jobStr = '_'.join(map(str, [ureg, ireg, usetbiasreg, gamma]))
              print 'cd ' + data + ' && ', setrec, NUSERS, NITEMS, 5, 5000, SEED, \
                  ureg, ireg, usetbiasreg, 0, 0, 0, \
                  LEARNRATE, isMixRat, gamma, trainSet, testSet, valSet, RATMAT, \
                  trainRatMat, TEST_RATMAT, VAL_RATMAT, prefix, \
                  " > " + opDir + "/" + prefix + "_" + str(trainMatInd)  + \
                  "_mix_" + jobStr + ".txt"


def main():
  setrec = sys.argv[1]
  data   = sys.argv[2]
  topParamFName = ''
  if len(sys.argv) > 3
    topParamFName = sys.argv[3];

  prefix = os.path.basename(setrec)
  
  for i in range(1,4):
    subDirName = 'buckets' + str(i)
    subDir = os.path.join(data, subDirName)
    if not os.path.exists(subDir):
      print 'Path not found: ', subDir
      return
    subPref = prefix# + '_' + subDirName
    genGridRegJobs(setrec, subPref, subDir)
    genMixJobsFromParam(setrec, subPref + 'mix', topParamFName, subDir)

  """
  for i in range(1,4):
    subDirName = 'var' + str(i)
    subDir = os.path.join(data, subDirName)
    if not os.path.exists(subDir):
      print 'Path not found: ', subDir
      return
    subPref = prefix# + '_' + subDirName
    genGridRegJobs(setrec, subPref, subDir)
    genMixJobsFromParam(setrec, subPref + 'mix', topParamFName, subDir)
    #genGridMixRegJobs(setrec, subPref + 'mix', subDir)
  """

if __name__ == '__main__':
  main()




