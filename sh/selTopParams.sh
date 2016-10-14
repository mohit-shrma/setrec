#!/bin/bash

ipFName=$1

prec=9

valItemRMSE=21
testItemRMSE=22
alltestRMSE=23

valSetRMSE=25
testSetRMSE=26

valSetBPR=27
testSetBPR=28
valItemBPR=29
testItemBPR=30
allAUC=31


sort -t' ' -gr -k$prec,$prec $ipFName |cut -d' ' -f2-8|head -25 

sort -t' ' -g -k$valItemRMSE,$valItemRMSE $ipFName |cut -d' ' -f2-8|head -25 
sort -t' ' -g -k$testItemRMSE,$testItemRMSE $ipFName |cut -d' ' -f2-8|head -25 
sort -t' ' -g -k$alltestRMSE,$alltestRMSE $ipFName |cut -d' ' -f2-8|head -25 

sort -t' ' -g -k$valSetRMSE,$valSetRMSE $ipFName |cut -d' ' -f2-8|head -25 
sort -t' ' -g -k$testSetRMSE,$testSetRMSE $ipFName |cut -d' ' -f2-8|head -25 

sort -t' ' -gr -k$valSetBPR,$valSetBPR $ipFName |cut -d' ' -f2-8|head -25 
sort -t' ' -gr -k$testSetBPR,$testSetBPR $ipFName |cut -d' ' -f2-8|head -25 

sort -t' ' -gr -k$valItemBPR,$valItemBPR $ipFName |cut -d' ' -f2-8|head -25 
sort -t' ' -gr -k$testItemBPR,$testItemBPR $ipFName |cut -d' ' -f2-8|head -25 
sort -t' ' -gr -k$allAUC,$allAUC $ipFName |cut -d' ' -f2-8|head -25 




