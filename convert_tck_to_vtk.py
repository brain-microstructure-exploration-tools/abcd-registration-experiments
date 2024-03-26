import argparse
import os
from pathlib import Path
import glob
import subprocess

parser = argparse.ArgumentParser(description='Converts a directory of tck files to vtk format')

parser.add_argument('in_dir', type=str, help='path to input directory with tck files')
parser.add_argument('out_dir', type=str, help='path to desired output directory for vtk files')

args = parser.parse_args()

in_dir = Path(os.path.abspath(args.in_dir))
out_dir = Path(os.path.abspath(args.out_dir))

if not out_dir.exists():
  os.mkdir(str(out_dir))

input_tck = glob.glob(str(in_dir) + '/*.tck')

if (len(input_tck) == 0):
  print("No tck files found at the specified input directory")
  quit()
  
for tck_file in input_tck:
  
  #print(tck_file)
  out_filename = '%s/%s.vtk' %(str(out_dir), Path(tck_file).stem)
  
  subprocess.run(['tckconvert', tck_file, out_filename, '-force'], stdout=subprocess.DEVNULL)
  
