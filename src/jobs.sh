#!/bin/bash

PARALLEL=$HOME/tools/parallel-20131022/src/parallel
SETREC=/home/grad02/mohit/exmoh/dev/bitbucket/setrec/src/setrec
IPFILE=/home/grad02/mohit/data/lastfm/lastfm-dataset-1K/last_user_6mon_artist_atleast50uFiltLabeledWkSetInd.ssv
OPDIR=/home/grad02/mohit/data/lastfm/lastfm-dataset-1K/lastOp  
#./setrec /home/grad02/mohit/data/lastfm/lastfm-dataset-1K/last_user_6mon_artist_atleast50uFiltLabeledWkSetInd.ssv 731 3087 5 0.001 0.001 0.001 1 10 1

userRegs=("0.1" "0.01" "0.001")
itemRegs=("0.1" "0.01" "0.001")
fDims=(1 5 10 15)
lRates=("0.001" "0.0001" "0.00001")

mkdir $OPDIR

for facDim in "${fDims[@]}";do
  for learnRate in "${lRates[@]}";do
    for uReg in "${userRegs[@]}";do
      for iReg in "${itemRegs[@]}";do
      echo $SETREC $IPFILE 730 3087 $facDim $uReg $iReg $learnRate "1 10000 1 > $OPDIR/setrec_"$facDim"_"$uReg"_"$iReg"_"$learnRate".txt"
      done
    done
  done
done



