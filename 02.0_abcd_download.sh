#!/bin/bash

# This needs the NDA download tool:
# `pip install nda-tools`. Then `downloadcmd --help` to see usage.

downloadcmd -t ~/abcd-registration-experiments/abcd-sample/sample_derived_files.txt -d ~/data/abcd/Package_1200530/ -u ebrahim -p '<put nda pw here>' -dp 1200530
