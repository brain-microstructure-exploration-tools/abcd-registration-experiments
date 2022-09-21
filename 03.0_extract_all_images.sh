#!/bin/bash
set -e

# The base directory to which the archive files were downloaded
ARCHIVES_PATH="/home/ebrahim/data/abcd/Package_1200530/fmriresults01/abcd-mproc-release4"

if [[ $# -ne 2 ]]; then
  echo "Usage: ./03.0_extract_all_images.sh  <DERIVED_FILE_LIST>  <DESTINATION_PATH>"
  echo "  where DERIVED_FILE_LIST is the file containing the list of S3 links, the same one you gave to NDA download manager,"
  echo "  and where DESTINATION_PATH is the directory to which you want to dump the extracted images."
  exit
fi

# The file containing the list of S3 links, the same one we gave to NDA download manager
DERIVED_FILE_LIST=$1

# The directory in which we want to put the extracted images
DESTINATION_PATH=$2


sed 's/.*\///' $DERIVED_FILE_LIST | while read ARCHIVE_NAME ; do
  A=$ARCHIVES_PATH/$ARCHIVE_NAME
  [ -f "$A" ] || break
  D=$DESTINATION_PATH/$(basename $A .tgz)
  if [ -d $D ]
  then
    continue
  fi
  mkdir $D
  tar -xvf $A --directory $D
done
