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

for dim in "${FACDIMS[@]}";do
  for reg in "${REGS[@]}";do
    echo $SETREC $NUSERS $NITEMS $dim 5000 $SEED $reg $reg 0.0 0.0 $LEARNRATE 0.0 0.0 \
      $TRAIN $TEST $VAL $RATMAT "avgwbias > setrec_avgwbias_"$reg"_"$dim".txt"
  done
done


