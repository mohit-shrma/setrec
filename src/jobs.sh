#!/bin/bash

SETREC=/home/grad02/mohit/data/movielens/ml-latest-small/majoritySet3/setrecMf
IPFILE=$1
OPDIR=/home/grad02/mohit/data/movielens/ml-latest-small/majoritySet3/matfacOp 
#./setrec /home/grad02/mohit/data/lastfm/lastfm-dataset-1K/last_user_6mon_artist_atleast50uFiltLabeledWkSetInd.ssv 731 3087 5 0.001 0.001 0.001 1 10 1

userRegs=("0.001" "0.01" "0.1" "1" )
itemRegs=("0.001" "0.01" "0.1" "1" )
umRegs=("0.01")
fDims=(1 5 10 15 30 50)
constWts=("1.0")
lRates=("0.001" "0.0001")
rRMSs=("0.95" "0.99")
splits=(1)
rm -rf $OPDIR
mkdir -p $OPDIR

for facDim in "${fDims[@]}";do
  for learnRate in "${lRates[@]}";do
    for uReg in "${userRegs[@]}";do
      for iReg in "${itemRegs[@]}";do
        for cWt in "${constWts[@]}";do
          for rRMS in "${rRMSs[@]}";do
            for umReg in "${umRegs[@]}";do
              for i in "${splits[@]}";do
                echo $SETREC $IPFILE"_split"$i 2059 28956 $facDim $uReg $iReg $cWt $learnRate \
                "0 10000 1"  \
                $IPFILE"_split"$i"_train.csr" \
                $IPFILE"_split"$i"_test.csr" $IPFILE"_split"$i"_val.csr" "null_"$i \ 
                $rRMS $umReg   "null null null  >" \
                $OPDIR/"setrec_"$facDim"_"$uReg"_"$iReg"_"$umReg"_"$cWt"_"$rRMS"_"$learnRate"_"$i".txt"
              done
            done
          done
        done
      done
    done
  done
done



