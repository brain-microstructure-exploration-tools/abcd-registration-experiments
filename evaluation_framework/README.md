This repository contains modules and driver/test scripts for evaluation of dwi image registration methods

# Requirements

Requirements will increase as more registration methods are added and more evaluation criteria are implemented.

numpy, vtk, nibel, hp5y, ANTsPy, mrtrix, dipy

# Data preparation

While several current driver scripts serve to test the methods starting from dwi, in general the evaluation framework will assume preprocessed data, including:

dti, fa images, brain masks, fod images, and tractography

# Run an example -- ants FA registration and evaluation via fiber tract distance

From inside the scripts directory, run the driver `pairwise_evaluation_ants.py`

```sh
python pairwise_evaluation_ants.py [-h] [--percent_sample_fibers PERCENT_SAMPLE_FIBERS] [--num_repeats NUM_REPEATS] [--force] source source_fiber_dir target target_fiber_dir output_base_dir exp_name
```

This will create a new directory `output_base_dir/exp_name/` to store evalutation results. The main result of interest for now is a csv file `fiber_distances.csv`.