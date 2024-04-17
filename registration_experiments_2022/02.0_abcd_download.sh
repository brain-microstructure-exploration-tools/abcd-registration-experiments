#!/bin/bash

# This needs the NDA download tool:
# `pip install nda-tools`. Then `downloadcmd --help` to see usage.

# Fill in your own username, and when prompted provide your own password
downloadcmd -t ~/abcd-registration-experiments/01.1_abcd_sample2/sample_all_derived_files.txt -d ~/data/abcd/Package_1200530/ -u ebrahim -dp 1200530
