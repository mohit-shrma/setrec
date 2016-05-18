#!/bin/bash
export LD_LIBRARY_PATH=$HOME/dev/gsl/lib
SETREC=""
DATA=""
NUSERS="854"
NITEMS="12549"
FACDIMS=(1 5 10 15 25)
REGS=("0.001" "0.01" "0.1" "1" "10")
LEARNRATE="0.001"
SEED="1"

TRAIN=$DATA"/"
TEST=$DATA"/"
VAL=$DATA"/"

RATMAT=$DATA"/"
TRAIN_RATMAT=$DATA"/"
TEST_RATMAT=$DATA"/"
VAL_RATMAT=$DATA"/"

for dim in "${FACDIMS[@]}";do
  for ireg in "${REGS[@]}";do
    for ureg in "${REGS[@]}";do
      for usetreg in "${REGS[@]}";do
        echo $SETREC $NUSERS $NITEMS $dim 5000 $SEED $ureg $ireg $usetreg 0.0 $LEARNRATE 0.0 0.0 \
          $TRAIN $TEST $VAL $RATMAT $TRAIN_RATMAT $TEST_RATMAT $VAL_RATMAT \
          "avgwsetbias > avgwsetbias_"$ureg"_"$ireg"_"$usetreg"_"$dim".txt"
      done
    done
  done
done


