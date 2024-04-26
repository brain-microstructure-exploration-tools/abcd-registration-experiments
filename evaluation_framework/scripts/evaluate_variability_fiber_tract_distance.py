import argparse
from pathlib import Path
import os
from timeit import default_timer as timer
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

import fiber_measures

# Arguments
parser = argparse.ArgumentParser(description='Evaluate variability of fiber tract distance to the sampling density of streamlines')

parser.add_argument('tract1', type=str, help='path to a fiber tract in tck format')
parser.add_argument('tract2', type=str, help='path to a fiber tract in tck format')
parser.add_argument('percent_sample_fibers',type=float, help='randomly sample a percentage of fiber streamlines')
parser.add_argument('num_repeats', type=int, help='the number of times to repeat the fiber tract distance measurement')
parser.add_argument('output_dir', type=str, help='path to a folder for saving output')
parser.add_argument('exp_name', type=str, help='a name for the experiment for saving the results')

args = parser.parse_args()

fiber_tract_M_path = args.tract1
fiber_tract_T_path = args.tract2
percent_sample_fibers = args.percent_sample_fibers
num_repeats = args.num_repeats
output_dir = Path(args.output_dir)
exp_name = args.exp_name

times = np.zeros((num_repeats, 1))
scores = np.zeros((num_repeats, 1))

# Create the output directory if it doesn't exist
if not output_dir.exists():
  os.mkdir(str(output_dir))  

# Filename for the csv file that will save the results
out_csv_file = '%s/%s.csv' %(output_dir, exp_name)
# The contents of the csv file
csv_contents = 'Score,Time'

# Repeat the tract distance a number of times
for i in range(0, num_repeats):
  
    start = timer()

    # Call the tract distance script
    scores[i] = fiber_measures.fiber_tract_distance(fiber_tract_M_path, fiber_tract_T_path, percent_sample_fibers)

    end = timer()
    elapsed_time = end - start

    times[i] = elapsed_time

    cur_row = '\n%f,%f' %(scores[i], times[i])
    csv_contents += cur_row

# Write csv
file = open(out_csv_file, "w") 
file.write(csv_contents)
file.close()

# Create a histogram and save it
out_score_hist_image = '%s/%s.png' %(output_dir, exp_name)
plot_title = 'Fiber tract distances for %0.2f fiber sampling (%d repeats)\nMean: %0.2f   Std dev: %0.2f   Avg runtime: %0.1f secs' %(percent_sample_fibers, num_repeats, np.mean(scores), np.std(scores), np.mean(times))

fig, ax = plt.subplots()
n, bins, patches = ax.hist(scores)
ax.set_title(plot_title)
ax.set_xlabel("Value")
ax.set_ylabel("Fiber tract distance")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

plt.savefig(out_score_hist_image)
