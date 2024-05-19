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
- [TractSeg](https://github.com/MIC-DKFZ/TractSeg)

# Data preparation

While several current driver scripts serve to test the methods starting from dwi, in general the evaluation framework will assume preprocessed data, including:

dti, fa images, brain masks, fod images, and tractography.

# Run an example -- ants FA registration and evaluation

## Data preprocessing

This example requires several preprocessing steps. We assume the user is starting from dwi images (with corresponding bval/bvec). Preprocessing consists of brain brasking, dti reconstruction, and tractography.

### Brain masking

### DTI reconstruction

### Subject-specific tractography

For this example, we will be using subject-specific tractography to quantify registration accuracy. This is done by applying the registration deformation field to the source fiber tracts and measuring the distance to the target fiber tracts. This requires independent tractography estimation in each indiviudal subject's space. In this step, we will estimate subject-specific full brain tractography.

#### Computation of fiber orientation distribution (fod) images.

Fod images can be estimated given dwi with mrtrix commmand [dti2fod](https://mrtrix.readthedocs.io/en/dev/reference/commands/dwi2fod.html)

#### Estimating tractography

Once a fod image has been computed, full brain tractography can be estimated using the script `single_subject_tractography.py`.

```sh 
python single_subject_tractograph.py /path/to/fod_image.nii.gz /path/to/output_directory
```
This will create a directory structure `tractseg_output/TOM_trackings` at the specified output directory. Inside this directory will be .tck files representing each tract.

## Running registration and evalutation

To proceed with the example, we assume the preprocessing has been performed for 2 subjects and we use the following organization to illustrate usage:

- a folder `/path/to/data/` with `source_fa.nii.gz` and `target_fa.nii.gz` fa images for the source and target subjects
- folders for each subject's tractography `/path/to/source/tractseg_output/TOM_trackings` and `/path/to/target/tractseg_output/TOM_trackings`
- folders for each subject's white matter tract binary segmentation masks `/path/to/source/tractseg_output/bundle_segmentations` and `/path/to/target/tractseg_output/bundle_segmentations`

Now you can run the driver `pairwise_evaluation_ants.py`

```sh
python pairwise_evaluation_ants.py  
       /path/to/data/source_fa.nii.gz  
       /path/to/source/tractseg_output/TOM_trackings /path/to/source/tractseg_output/bundle_segmentations  
       /path/to/data/target_fa.nii.gz
       /path/to/target/tractseg_output/TOM_trackings /path/to/target/tractseg_output/bundle_segmentations  
       /path/to/output_base_directory my_test_exp
```

This will create a new directory `/path/to/output_base_directory/my_test_exp/` to store evalutation results. The main results are csv files in directory `evaluation_measures` named `fiber_measures.csv`, `segmentation_measures.csv`, and `transformation_measures.csv`. Experiment metadata are stored in a json file `my_test_exp.json`.  
