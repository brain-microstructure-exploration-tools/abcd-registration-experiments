This repository contains modules and driver/test scripts for evaluation of dwi image registration methods.

# Requirements

Requirements will increase as more registration methods are added and more evaluation criteria are implemented.

Python version 3.10.14 and the following packages: 

- [antspyx (for ANTsPy)](https://github.com/ANTsX/ANTsPy)
- [dipy](https://dipy.org/) 
- [h5py](https://github.com/h5py/h5py)
- [nibabel](https://nipy.org/nibabel/)
- [numpy](https://numpy.org/)
- [vtk](https://pypi.org/project/vtk/)

Addtional sofware:
- [mrtrix](https://www.mrtrix.org/) Tested with version 3.0.3 installed via conda.


# Data preparation

While several current driver scripts serve to test the methods starting from dwi, in general the evaluation framework will assume preprocessed data, including:

dti, fa images, brain masks, fod images, and tractography.

# Run an example -- ants FA registration and evaluation via fiber tract distance

From inside the scripts directory, run the driver `pairwise_evaluation_ants.py`

```sh
python pairwise_evaluation_ants.py [-h] [--percent_sample_fibers PERCENT_SAMPLE_FIBERS] [--num_repeats NUM_REPEATS] source source_mask source_fiber_dir target target_mask target_fiber_dir output_dir
```

This will create a new directory `output_dir` to store evalutation results. The main result of interest for now is inside folder `fiber_distances`.
