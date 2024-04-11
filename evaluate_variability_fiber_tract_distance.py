import argparse
from pathlib import Path
import subprocess
import os
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# Arguments
parser = argparse.ArgumentParser(description='Evaluate variability of fiber tract score to the sampling density of streamlines')

parser.add_argument('bundle1', type=str, help='path to a fiber bundle')
parser.add_argument('bundle2', type=str, help='path to a fiber bundle')
parser.add_argument('percent_sample_fibers',type=float, help='Randomly sample a percentage of fiber streamlines')
parser.add_argument('num_repeats', type=int, help='the number of times to repeat the fiber tract distance measurement')
parser.add_argument('output_dir', type=str, help='path to a folder for saving output')
parser.add_argument('exp_name', type=str, help='a name for the experiment for saving the results')

args = parser.parse_args()

fiber_bundle_M_path = args.bundle1
fiber_bundle_T_path = args.bundle2
percent_sample_fibers = args.percent_sample_fibers
num_repeats = args.num_repeats
output_dir = Path(args.output_dir)
exp_name = args.exp_name

timeings = np.zeros((num_repeats, 1))
scores = np.zeros((num_repeats, 1))

# Create the output directory if it doesn't exist
if not output_dir.exists():
  os.mkdir(str(output_dir))  

# The path to compute tract distance
script_path = Path(os.path.realpath(__file__))
script_location = '%s/compute_tract_distance.py' %(str(script_path.parent))

# Filename for the csv file that will save the results
out_csv_file = '%s/%s.csv' %(output_dir, exp_name)
# The contents of the csv file
csv_contents = 'Score,Time'

# Repeat the tract distance a number of times
for i in range(0, num_repeats):
  
    start = timer()

    # Call the tract distance script
    result = subprocess.run(['python', script_location, '--percent_sample_fibers', str(percent_sample_fibers), fiber_bundle_M_path, fiber_bundle_T_path], capture_output = True, text = True)

    end = timer()
    elapsed_time = end - start

    scores[i] = float(result.stdout)
    timeings[i] = elapsed_time

    cur_row = '\n%f,%f' %(scores[i], timeings[i])
    csv_contents += cur_row

# Write csv
file = open(out_csv_file, "w") 
file.write(csv_contents)
file.close()

# Create a histogram and save it
out_score_hist_image = '%s/%s.png' %(output_dir, exp_name)
plot_title = 'Fiber tract distances for %0.2f fiber sampling (%d repeats)\nMean: %0.2f   Std dev: %0.2f   Avg runtime: %0.1f secs' %(percent_sample_fibers, num_repeats, np.mean(scores), np.std(scores), np.mean(timeings))

fig, ax = plt.subplots()
n, bins, patches = ax.hist(scores)
ax.set_title(plot_title)
ax.set_xlabel("Value")
ax.set_ylabel("Fiber tract distance")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig(out_score_hist_image)