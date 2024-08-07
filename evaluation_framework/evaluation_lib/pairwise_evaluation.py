import os
import subprocess
from pathlib import Path
from tempfile import TemporaryDirectory

import nibabel as nib

from evaluation_lib import (fiber_measures, segmentation_measures,
                            transformation_measures, transformation_utils)

### Constants for output directories

WARPED_FIBERS_DIR = 'warped_fibers'
WARPED_SEGMENTATION_DIR = 'warped_segmentations'
EVALUATION_MEASURES_DIR = 'evaluation_measures'

### Define some constants which define which fiber tracts to oinclude in the evaluation ###

DEFAULT_FIBER_TRACTS = ['AF_left.tck', 'AF_right.tck', 'ATR_left.tck', 'ATR_right.tck', 'CA.tck', 'CC.tck', \
                        'CC_1.tck', 'CC_2.tck', 'CC_3.tck', 'CC_4.tck', 'CC_5.tck', 'CC_6.tck', 'CC_7.tck', \
                        'CG_left.tck', 'CG_right.tck', 'CST_left.tck', 'CST_right.tck', 'FPT_left.tck', 'FPT_right.tck', \
                        'FX_left.tck', 'FX_right.tck', 'ICP_left.tck', 'ICP_right.tck', 'IFO_left.tck', 'IFO_right.tck', \
                        'ILF_left.tck', 'ILF_right.tck', 'MCP.tck', 'MLF_left.tck', 'MLF_right.tck', 'OR_left.tck', 'OR_right.tck', \
                        'POPT_left.tck', 'POPT_right.tck', 'SCP_left.tck', 'SCP_right.tck', 'SLF_III_left.tck', 'SLF_III_right.tck', \
                        'SLF_II_left.tck', 'SLF_II_right.tck', 'SLF_I_left.tck', 'SLF_I_right.tck', 'STR_left.tck', 'STR_right.tck', \
                        'ST_FO_left.tck', 'ST_FO_right.tck', 'ST_OCC_left.tck', 'ST_OCC_right.tck', 'ST_PAR_left.tck', 'ST_PAR_right.tck', \
                        'ST_POSTC_left.tck', 'ST_POSTC_right.tck', 'ST_PREC_left.tck', 'ST_PREC_right.tck', 'ST_PREF_left.tck', 'ST_PREF_right.tck', \
                        'ST_PREM_left.tck', 'ST_PREM_right.tck', 'T_OCC_left.tck', 'T_OCC_right.tck', 'T_PAR_left.tck', 'T_PAR_right.tck', \
                        'T_POSTC_left.tck', 'T_POSTC_right.tck', 'T_PREC_left.tck', 'T_PREC_right.tck', 'T_PREF_left.tck', 'T_PREF_right.tck', \
                        'T_PREM_left.tck', 'T_PREM_right.tck', 'UF_left.tck', 'UF_right.tck']

TESTING_FIBER_TRACTS = ['AF_left.tck', 'AF_right.tck', 'ATR_left.tck', 'ATR_right.tck']

DEFAULT_SEGMENTATIONS = ['AF_left.nii.gz', 'AF_right.nii.gz', 'ATR_left.nii.gz', 'ATR_right.nii.gz', 'CA.nii.gz', 'CC.nii.gz', \
                        'CC_1.nii.gz', 'CC_2.nii.gz', 'CC_3.nii.gz', 'CC_4.nii.gz', 'CC_5.nii.gz', 'CC_6.nii.gz', 'CC_7.nii.gz', \
                        'CG_left.nii.gz', 'CG_right.nii.gz', 'CST_left.nii.gz', 'CST_right.nii.gz', 'FPT_left.nii.gz', 'FPT_right.nii.gz', \
                        'FX_left.nii.gz', 'FX_right.nii.gz', 'ICP_left.nii.gz', 'ICP_right.nii.gz', 'IFO_left.nii.gz', 'IFO_right.nii.gz', \
                        'ILF_left.nii.gz', 'ILF_right.nii.gz', 'MCP.nii.gz', 'MLF_left.nii.gz', 'MLF_right.nii.gz', 'OR_left.nii.gz', 'OR_right.nii.gz', \
                        'POPT_left.nii.gz', 'POPT_right.nii.gz', 'SCP_left.nii.gz', 'SCP_right.nii.gz', 'SLF_III_left.nii.gz', 'SLF_III_right.nii.gz', \
                        'SLF_II_left.nii.gz', 'SLF_II_right.nii.gz', 'SLF_I_left.nii.gz', 'SLF_I_right.nii.gz', 'STR_left.nii.gz', 'STR_right.nii.gz', \
                        'ST_FO_left.nii.gz', 'ST_FO_right.nii.gz', 'ST_OCC_left.nii.gz', 'ST_OCC_right.nii.gz', 'ST_PAR_left.nii.gz', 'ST_PAR_right.nii.gz', \
                        'ST_POSTC_left.nii.gz', 'ST_POSTC_right.nii.gz', 'ST_PREC_left.nii.gz', 'ST_PREC_right.nii.gz', 'ST_PREF_left.nii.gz', 'ST_PREF_right.nii.gz', \
                        'ST_PREM_left.nii.gz', 'ST_PREM_right.nii.gz', 'T_OCC_left.nii.gz', 'T_OCC_right.nii.gz', 'T_PAR_left.nii.gz', 'T_PAR_right.nii.gz', \
                        'T_POSTC_left.nii.gz', 'T_POSTC_right.nii.gz', 'T_PREC_left.nii.gz', 'T_PREC_right.nii.gz', 'T_PREF_left.nii.gz', 'T_PREF_right.nii.gz', \
                        'T_PREM_left.nii.gz', 'T_PREM_right.nii.gz', 'UF_left.nii.gz', 'UF_right.nii.gz']

TESTING_SEGMENTATIONS = ['AF_left.nii.gz', 'AF_right.nii.gz', 'ATR_left.nii.gz', 'ATR_right.nii.gz']

def pairwise_evaluation_ants(
    target_fa_path: Path, forward_diffeo_path: Path, inverse_diffeo_path: Path, 
    source_fiber_path: Path, source_segmentation_path: Path,
    target_fiber_path: Path, target_segmentation_path: Path,
    output_path: Path, percent_sample_fibers: float=0.1, num_repeats: int=1, specified_fibers: list=[], specified_segmentations: list=[]
) -> None:
    """
    Performs a pairwise registration evalutation given the output of ants registration and directories for source and target fiber tracts and segmentations
    
    :param target_fa_path: the target fa image in .nii.gz format
    :param forward_diffeo_path: the forward diffeo from ants registration
    :param inverse_diffeo_path: the inverse diffeo from ants registration
    :param source_fiber_path: a directory with source fibers in tck format
    :param source_segmentation_path: a directory with source binary segmentations in nii.gz format
    :param target_fiber_path: a directory with target fibers in tck format which correspond by name to source fibers
    :param target_segmentation_path: a directory with target binary segmentations in nii.gz format
    :param output_path: base directory to write output
    :param percent_sample_fibers: a percentage of fiber streamlines to sample when computing fiber tract distance
    :param num_repeats: the number of times to compute fiber tract distance with different random sampling
    :param specified_fibers: a list of strings specifying which fibers to use
    :param specified_segmentations: a list of strings specifying which segmentations to use
    """

    ### Convert the transformations to mrtrix format ###

    with TemporaryDirectory() as temp_dir:    

        mrtrix_forward_warp = transformation_utils.convert_ants_transform_to_mrtrix_transform(target_fa_path, forward_diffeo_path)
        mrtrix_inverse_warp = transformation_utils.convert_ants_transform_to_mrtrix_transform(target_fa_path, inverse_diffeo_path)

        out_forward_warp_filename = Path(temp_dir) / 'mrtrix_forward_diffeo.nii.gz'
        out_inverse_warp_filename = Path(temp_dir) / 'mrtrix_inverse_diffeo.nii.gz'

        nib.save(mrtrix_forward_warp, out_forward_warp_filename)
        nib.save(mrtrix_inverse_warp, out_inverse_warp_filename)

        ### Compute metrics ###

        compute_fiber_metrics(out_inverse_warp_filename, source_fiber_path, target_fiber_path, output_path, percent_sample_fibers, num_repeats, specified_fibers)
        compute_segmentation_metrics(out_forward_warp_filename, source_segmentation_path, target_segmentation_path, output_path, specified_segmentations)
        compute_transformation_metrics(out_forward_warp_filename, out_inverse_warp_filename, output_path)

def pairwise_evaluation_voxelmorph(
    target_fa_path: Path, forward_diffeo_path: Path, inverse_diffeo_path: Path, 
    source_fiber_path: Path, source_segmentation_path: Path,
    target_fiber_path: Path, target_segmentation_path: Path,
    output_path: Path, percent_sample_fibers: float=0.1, num_repeats: int=1, specified_fibers: list=[], specified_segmentations: list=[], use_gpu: bool=False
) -> None:
    """
    Performs a pairwise registration evalutation given the output of voxelmorph registration and directories for source and target fiber tracts and segmentations
    
    :param target_fa_path: the target fa image in .nii.gz format
    :param forward_diffeo_path: the forward diffeo from voxelmorph registration
    :param inverse_diffeo_path: the inverse diffeo from voxelmorph registration (currently not used/implemented)
    :param source_fiber_path: a directory with source fibers in tck format
    :param source_segmentation_path: a directory with source binary segmentations in nii.gz format
    :param target_fiber_path: a directory with target fibers in tck format which correspond by name to source fibers
    :param target_segmentation_path: a directory with target binary segmentations in nii.gz format
    :param output_path: base directory to write output
    :param percent_sample_fibers: a percentage of fiber streamlines to sample when computing fiber tract distance
    :param num_repeats: the number of times to compute fiber tract distance with different random sampling
    :param specified_fibers: a list of strings specifying which fibers to use
    :param specified_segmentations: a list of strings specifying which segmentations to use
    :param use_gpu: use the gpu or cpu
    """

    ### Convert the transformations to mrtrix format ###

    with TemporaryDirectory() as temp_dir:    

        # Convert voxelmorph transformation to mrtrix
        mrtrix_forward_warp = transformation_utils.convert_voxelmorph_transform_to_mrtrix_transform(target_fa_path, forward_diffeo_path, use_gpu=use_gpu)
        out_forward_warp_filename = Path(temp_dir) / 'mrtrix_forward_diffeo.nii.gz'
        nib.save(mrtrix_forward_warp, out_forward_warp_filename)
        
        ## Compute the inverse mrtrix warp using the command warpinvert (currently do not know how to obtain inverse from voxelmorph)
        out_inverse_warp_filename = Path(temp_dir) / 'mrtrix_inverse_diffeo.nii.gz'
        subprocess.run(['warpinvert', str(out_forward_warp_filename), out_inverse_warp_filename, '-force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        ### Compute metrics ###

        compute_fiber_metrics(out_inverse_warp_filename, source_fiber_path, target_fiber_path, output_path, percent_sample_fibers, num_repeats, specified_fibers)
        compute_segmentation_metrics(out_forward_warp_filename, source_segmentation_path, target_segmentation_path, output_path, specified_segmentations)
        compute_transformation_metrics(out_forward_warp_filename, out_inverse_warp_filename, output_path)

def pairwise_evaluation_dtitk(forward_diffeo_path: Path, inverse_diffeo_path: Path, 
    source_fiber_path: Path, source_segmentation_path: Path,
    target_fiber_path: Path, target_segmentation_path: Path,
    output_path: Path, percent_sample_fibers: float=0.1, num_repeats: int=1, specified_fibers: list=[], specified_segmentations: list=[]
) -> None:
    """
    Performs a pairwise registration evalutation given the output of dtitk registration and directories for source and target fiber tracts and segmentations
    
    :param forward_diffeo_path: the forward diffeo from voxelmorph registration
    :param inverse_diffeo_path: the inverse diffeo from voxelmorph registration (currently not used/implemented)
    :param source_fiber_path: a directory with source fibers in tck format
    :param source_segmentation_path: a directory with source binary segmentations in nii.gz format
    :param target_fiber_path: a directory with target fibers in tck format which correspond by name to source fibers
    :param target_segmentation_path: a directory with target binary segmentations in nii.gz format
    :param output_path: base directory to write output
    :param percent_sample_fibers: a percentage of fiber streamlines to sample when computing fiber tract distance
    :param num_repeats: the number of times to compute fiber tract distance with different random sampling
    :param specified_fibers: a list of strings specifying which fibers to use
    :param specified_segmentations: a list of strings specifying which segmentations to use
    """

    ### Convert the transformations to mrtrix format ###

    with TemporaryDirectory() as temp_dir:    

        # Convert voxelmorph transformation to mrtrix
        #mrtrix_forward_warp = transformation_utils.convert_voxelmorph_transform_to_mrtrix_transform(target_fa_path, forward_diffeo_path, gpu=gpu)
        out_forward_warp_filename = Path(temp_dir) / 'mrtrix_forward_diffeo.nii.gz'
        subprocess.run(['warpconvert',forward_diffeo_path, 'displacement2deformation', out_forward_warp_filename, '-force'])
        #nib.save(mrtrix_forward_warp, out_forward_warp_filename)
        
        ## Compute the inverse mrtrix warp using the command warpinvert (currently do not know how to obtain inverse from voxelmorph)
        out_inverse_warp_filename = Path(temp_dir) / 'mrtrix_inverse_diffeo.nii.gz'
        subprocess.run(['warpinvert', str(out_forward_warp_filename), out_inverse_warp_filename, '-force'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        
        ### Compute metrics ###

        compute_fiber_metrics(out_inverse_warp_filename, source_fiber_path, target_fiber_path, output_path, percent_sample_fibers, num_repeats, specified_fibers)
        compute_segmentation_metrics(out_forward_warp_filename, source_segmentation_path, target_segmentation_path, output_path, specified_segmentations)
        compute_transformation_metrics(out_forward_warp_filename, out_inverse_warp_filename, output_path)

def compute_fiber_metrics(
    diffeo_path: Path, source_fiber_path: Path, target_fiber_path: Path, output_path: Path, 
    percent_sample_fibers: float, num_repeats: int, specified_fibers: list=[]
    ) -> None:
    """
    Computes fiber based metrics (currently only fiber tract distance) and outputs a csv to the specified output path
    
    :param diffeo_path: the diffeo used to warp source fibers
    :param source_fiber_path: a directory with source fibers in tck format
    :param target_fiber_path: a directory with target fibers in tck format which correspond by name to source fibers
    :param output_path: base directory to write output
    :param percent_sample_fibers: a percentage of fiber streamlines to sample when computing fiber tract distance
    :param num_repeats: the number of times to compute fiber tract distance with different random sampling
    :param specified_fibers: a list of strings specifying which fibers to use
    """

    # If the user did not specify which subset of fibers to use, we will use the "defaul set" (which we can determine later -- right now its all tracts)
    if (len(specified_fibers) == 0):

        specified_fibers = DEFAULT_FIBER_TRACTS    

    # Output folders for warped fibers
    fiber_out_path = output_path / WARPED_FIBERS_DIR

    if not fiber_out_path.exists():
        os.mkdir(str(fiber_out_path))

    # Measures output path 
    evaluation_measures_output_path = output_path / EVALUATION_MEASURES_DIR 
    
    if not evaluation_measures_output_path.exists():
        os.mkdir(str(evaluation_measures_output_path))

    fiber_distance_csv_path = evaluation_measures_output_path / 'fiber_measures.csv'
    fiber_distance_csv = 'Fiber Tract Name, Fiber Tract Distance\n'

    for i in range(0, len(specified_fibers)):

        cur_fiber_tract = source_fiber_path / specified_fibers[i]
        target_fiber_tract = target_fiber_path / specified_fibers[i]

        # If it's not there, skip it
        if (not cur_fiber_tract.exists() or not target_fiber_tract.exists()):
            continue

        warped_fiber_path = fiber_out_path / f'{specified_fibers[i]}_warped.tck'

        # Deform the source fiber tract
        transformation_utils.warp_fiber_tract(cur_fiber_tract, diffeo_path, warped_fiber_path)

        # Compute fiber tract distance
        cur_fiber_dist = fiber_measures.fiber_tract_distance(warped_fiber_path, target_fiber_tract, percent_sample_fibers, num_repeats)
        
        fiber_distance_csv += f'{specified_fibers[i]}, {str(cur_fiber_dist)}\n'

    # Write fiber distance csv
    f = open(str(fiber_distance_csv_path), 'w')
    f.write(fiber_distance_csv)
    f.close()

def compute_segmentation_metrics(diffeo_path: Path, source_segmentation_path: Path, target_segmentation_path: Path, output_path: Path, specified_segmentations: list=[]) -> None:
    """
    Computes segmentation based metrics (currently dice and volumetric similarity) and outputs a csv to the specified output path
    
    :param diffeo_path: the diffeo used to warp source segmentations
    :param source_segmentation_path: a directory with source binary segmentations in nii.gz format
    :param target_segmentation_path: a directory with target binary segmentations in nii.gz format
    :param output_path: base directory to write output
    :param specified_segmentations: a list of strings specifying which segmentations to use
    """

    # If the user did not specify which subset of segmenations to use, we will use the "defaul set"
    if (len(specified_segmentations) == 0):

        specified_segmentations = DEFAULT_SEGMENTATIONS

    # Output folders for warped fibers and for fiber tract distances
    segmentation_out_path = output_path / WARPED_SEGMENTATION_DIR

    if not segmentation_out_path.exists():
        os.mkdir(str(segmentation_out_path))

    # Measures output path 
    evaluation_measures_output_path = output_path / EVALUATION_MEASURES_DIR 
    
    if not evaluation_measures_output_path.exists():
        os.mkdir(str(evaluation_measures_output_path))

    segmentation_measures_csv_path = evaluation_measures_output_path / 'segmentation_measures.csv'
    segmentation_measures_csv = 'Segmentation Name, Dice, Volumetric Similarity\n'

    for i in range(0, len(specified_segmentations)):

        cur_segmentation = source_segmentation_path / specified_segmentations[i]
        target_segmentation = target_segmentation_path / specified_segmentations[i]

        # If it's not there, skip it
        if (not cur_segmentation.exists() or not target_segmentation.exists()):
            continue

        warped_segmentation_path = segmentation_out_path / f'{specified_segmentations[i]}_warped.nii.gz'

        # Deform the source segmentation image
        transformation_utils.warp_segmentation_image(cur_segmentation, diffeo_path, warped_segmentation_path)

        dice = segmentation_measures.dice_overlap(warped_segmentation_path, target_segmentation)
        volumetric_similarity = segmentation_measures.volumetric_similarity(warped_segmentation_path, target_segmentation)
        
        segmentation_measures_csv += f'{specified_segmentations[i]}, {dice}, {volumetric_similarity}\n'

    # Write segmentation measures csv
    f = open(str(segmentation_measures_csv_path), 'w')
    f.write(segmentation_measures_csv)
    f.close()

def compute_transformation_metrics(forward_diffeo_path: Path, inverse_diffeo_path: Path, output_path: Path) -> None:

    """
    Computes transformation based metrics (currently percentage of voxels with negative jacobian determinant) and outputs a csv to the specified base output path
    
    :param forward_diffeo_path: the forward diffeo used in mrtrix format
    :param inverse_diffeo_path: the forward diffeo used in mrtrix format
    :param output_path: base directory to write output
    """

    # Measures output path 
    evaluation_measures_output_path = output_path / EVALUATION_MEASURES_DIR 
    
    if not evaluation_measures_output_path.exists():
        os.mkdir(str(evaluation_measures_output_path))

    transformation_measures_csv_path = evaluation_measures_output_path / 'transformation_measures.csv'
    transfromation_measures_csv = 'Transformation, Percent Negative Jacobian Determinant\n'

    transfromation_measures_csv += f'forward_diffeo, {transformation_measures.percent_negative_jacobian_determinant(forward_diffeo_path)}\n'
    transfromation_measures_csv += f'inverse_diffeo, {transformation_measures.percent_negative_jacobian_determinant(inverse_diffeo_path)}\n'
    
    # Write segmentation measures csv
    f = open(str(transformation_measures_csv_path), 'w')
    f.write(transfromation_measures_csv)
    f.close()