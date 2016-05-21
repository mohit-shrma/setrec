import random

NUSERS    = 854
NITEMS    = 12549
FACDIMS   = [1, 5, 10, 15, 25, 50, 75]
REGS      = [0.001, 0.01, 0.1, 1, 10]
LEARNRATE = 0.001
SEED      = 1

SETREC       = ""
DATA         = "/"

TRAIN_SET    = DATA + ""
TEST_SET     = DATA + ""
VAL_SET      = DATA + ""

RATMAT       = DATA + ""
TRAIN_RATMAT = DATA + ""
TEST_RATMAT  = DATA + ""
VAL_RATMAT   = DATA + ""

PREFIX       = ""

OPDIR        = "" 

for dim in FACDIMS:
  nJobs       = 0
  while nJobs < 100:
    ureg        = REGS[random.randint(0, len(REGS)-1)]
    ireg        = REGS[random.randint(0, len(REGS)-1)]
    ubiasreg    = REGS[random.randint(0, len(REGS)-1)]
    ibiasreg    = REGS[random.randint(0, len(REGS)-1)]
    usetbiasreg = REGS[random.randint(0, len(REGS)-1)]
    jobStr      = '_'.join(map(str, [ureg, ireg, ubiasreg, ibiasreg, usetbiasreg]))
    print SETREC, NUSERS, NITEMS, dim, 5000, SEED, \
        ureg, ireg, usetbiasreg, ubiasreg, ibiasreg, 0, LEARNRATE, 0, 0, \
        TRAIN_SET, TEST_SET, VAL_SET, RATMAT, TRAIN_RATMAT, TEST_RATMAT, \
        VAL_RATMAT, PREFIX, \
        " > " + OPDIR + "/" + PREFIX + "_" + jobStr + ".txt"

    nJobs += 1
  

