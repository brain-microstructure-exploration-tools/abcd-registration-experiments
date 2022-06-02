#!/bin/bash

ARCHIVES_PATH="/home/ebrahim/data/abcd/Package_1200530/fmriresults01/abcd-mproc-release4"
DESTINATION_PATH="/home/ebrahim/data/abcd/DMRI_extracted"

set -e

for A in $ARCHIVES_PATH/*.tgz; do
  [ -f "$A" ] || break
  D=$DESTINATION_PATH/$(basename $A .tgz)
  if [ -d $D ]
  then
    continue
  fi
  mkdir $D
  tar -xvf $A --directory $D
done
