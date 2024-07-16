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
- [voxelmorph](https://github.com/voxelmorph/voxelmorph)
- [tensorflow](https://www.tensorflow.org/) Tested with version 2.10.0

Addtional sofware:
- [mrtrix](https://www.mrtrix.org/) Tested with version 3.0.3 installed via conda.
- [TractSeg](https://github.com/MIC-DKFZ/TractSeg)
- [DTI-TK](https://dti-tk.sourceforge.net/pmwiki/pmwiki.php) Tested with version 2.3.1.
- [convert3D](http://www.itksnap.org/pmwiki/pmwiki.php?n=Convert3D.Convert3D) Tested with nightly version 2024-03-12. 

# Data preparation

While several current driver scripts serve to test the methods starting from dwi, in general the evaluation framework will assume preprocessed data, including:

dti, fa images, brain masks, fod images, and tractography.

# Run an example -- ants FA registration and evaluation

## Data preprocessing

This example requires several preprocessing steps. Starting from dwi images (with corresponding bval/bvec), use the pipeline in [abcd-noddi-tbss-experiments](https://github.com/brain-microstructure-exploration-tools/abcd-noddi-tbss-experiments) to generate brain masks, dti, and FODs.

### Subject-specific tractography

We will be using subject-specific tractography to quantify registration accuracy. This is done by applying the registration deformation field to the source fiber tracts and measuring the distance to the target fiber tracts. This requires independent tractography estimation in each indiviudal subject's space.

The pipeline in [abcd-noddi-tbss-experiments](https://github.com/brain-microstructure-exploration-tools/abcd-noddi-tbss-experiments) has a tractography step whose output can be used. If instead we want to run subject-specific full brain tractography for a single subject, then we also have the script `single_subject_tractography.py`.

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


# Voxelmorph evaluation

Download one of the pretrained models [here](https://github.com/voxelmorph/voxelmorph/blob/dev/data/readme.md).

Voxelmorph is tricky because it requires a certain version of tensorflow and that version of tensorflow requires a certain version of CUDA to run things on GPU. We ned to run voxelmorph on GPU in order to fairly evaluate the runtime when it's used as intended. Here we recommend setting up the nvidia container toolkit on your system and using the Docker image that we include.

From the present directory, and having install docker and nvidia container toolkit, run the following to build the image:
```
# For python3.9:
docker build -f voxelmorph.dockerfile -t voxelmorph-image .

# Or for python3.8:
docker build -f voxelmorph_python3.8.dockerfile -t voxelmorph-image .
```

Test that the image works:
```
docker run --gpus all --rm --name voxelmorph-container -v $(pwd):/workspace -u $(id -u):$(id -g) voxelmorph-image \
    python3.9 pairwise_evaluation_voxelmorph.py --help

# (Or if you built the python3.8 image replace "python3.9" by "python3.8")
```
A few harmless warnings and a usage text hopefully show up.

In order to use `pairwise_evaluation_voxelmorph.py`, run the above docker command with `python3.9 pairwise_evaluation_voxelmorph.py --help` being replaced by your desired way of calling `pairwise_evaluation_voxelmorph.py`.

# DTI-TK evaluation

- Download and extract DTI-TK and add the `bin/` and `scripts/` subdirectories to the `PATH` environment variable
- Also set the environment variable `DTITK_ROOT` to be the path to the extracted DTI-TK (the parent directory of `bin/` and `scripts`)
- Download and extract convert3d and add the `bin`/ subdirectory to the `PATH`

While this runs, it doesn't appear to be working well. Results look oddly deformed and not quite aligned. It is not clear whether it's an issue with DTI-TK or an issue with the way we are using it here.