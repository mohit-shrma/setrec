#!/bin/bash

ipFName=$1

prec=8

valItemRMSE=13
testItemRMSE=14
alltestRMSE=15

valSetRMSE=17
testSetRMSE=18

valSetBPR=19
testSetBPR=20
valItemBPR=21
testItemBPR=22
allAUC=23


sort -t' ' -gr -k$prec,$prec $ipFName |cut -d' ' -f2-7|head -20 
#sort -t' ' -gr -k$prec,$prec $ipFName |cut -d' ' -f2-6,8|head -20 

#sort -t' ' -g -k$valItemRMSE,$valItemRMSE $ipFName |cut -d' ' -f2-7|head -20 
#sort -t' ' -g -k$testItemRMSE,$testItemRMSE $ipFName |cut -d' ' -f2-7|head -20 
#sort -t' ' -g -k$alltestRMSE,$alltestRMSE $ipFName |cut -d' ' -f2-7|head -20 

#sort -t' ' -g -k$valSetRMSE,$valSetRMSE $ipFName |cut -d' ' -f2-7|head -20 
#sort -t' ' -g -k$testSetRMSE,$testSetRMSE $ipFName |cut -d' ' -f2-7|head -20 

sort -t' ' -gr -k$valSetBPR,$valSetBPR $ipFName |cut -d' ' -f2-7|head -20 
sort -t' ' -gr -k$testSetBPR,$testSetBPR $ipFName |cut -d' ' -f2-7|head -20 
sort -t' ' -gr -k$valItemBPR,$valItemBPR $ipFName |cut -d' ' -f2-7|head -20 
sort -t' ' -gr -k$testItemBPR,$testItemBPR $ipFName |cut -d' ' -f2-7|head -20 
sort -t' ' -gr -k$allAUC,$allAUC $ipFName |cut -d' ' -f2-7|head -20 




