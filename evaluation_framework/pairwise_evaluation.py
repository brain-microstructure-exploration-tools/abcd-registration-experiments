from pathlib import Path
import os
import glob

import ants
import nibabel as nib

import transformation_utils
import fiber_measures

def pairwise_evaluation_ants(
    target_fa_path: Path, forward_diffeo_path: Path, inverse_diffeo_path: Path, 
    source_fiber_path: Path, target_fiber_path: Path, 
    output_path: Path, percent_sample_fibers: float=0.1, num_repeats: int=1
) -> None:
    """
    Performs a pairwise registration evalutation given the output of ants registration and a directories source and target fiber tracts
    Currently this outputs transformations in mrtrix format, a folder of warped source fiber tracts, and a folder of scores for each fiber tract
    (This will be expanded to compute additional scores and may output something like a csv file to summarize all the scores)
    
    :param target_fa_path: the target fa image in .nii.gz format
    :param forward_diffeo_path: the forward diffeo from ants registration (not currently used in scoring)
    :param inverse_diffeo_path: the inverse diffeo from ants registration
    :param source_fiber_path: a directory with source fibers in tck format (may ultimate give an option to choose a subset of tracts within folder)
    :param target_fiber_path: a directory with target fibers in tck format which correspond by name to source fibers
    :param output_path: base directory to write output
    :param percent_sample_fibers: a percentage of fiber streamlines to sample when computing fiber tract distance
    :param num_repeats: the number of times to compute fiber tract distance with different random sampling
    """

    ### Convert the transformations to mrtrix format ###

    mrtrix_forward_warp = transformation_utils.convert_ants_transform_to_mrtrix_transform(target_fa_path, forward_diffeo_path)
    mrtrix_inverse_warp = transformation_utils.convert_ants_transform_to_mrtrix_transform(target_fa_path, inverse_diffeo_path)

    out_forward_warp_filename = '%s/mrtrix_forward_diffeo.nii.gz' %(str(output_path))
    out_inverse_warp_filename = '%s/mrtrix_inverse_diffeo.nii.gz' %(str(output_path))

    nib.save(mrtrix_forward_warp, out_forward_warp_filename)
    nib.save(mrtrix_inverse_warp, out_inverse_warp_filename)

    ### Measure fiber distance ### (This is a target for refactoring to a seperate module as we add additional scores, all scores could be organized in a csv or similar)

    # Assumes the same number of fibers with the corresponding names
    all_source_fibers = sorted(glob.glob(str(source_fiber_path) + '/*.tck'))
    all_target_fibers = sorted(glob.glob(str(target_fiber_path) + '/*.tck'))

    # Output folders for warped fibers and for fiber tract distances
    fiber_out_path = Path('%s/warped_fibers' %(str(output_path)))
    fiber_distance_path = Path('%s/fiber_distances' %(str(output_path)))

    if not fiber_out_path.exists():
        os.mkdir(str(fiber_out_path))

    if not fiber_distance_path.exists():
        os.mkdir(str(fiber_distance_path))

    for i in range(0, len(all_source_fibers)):
 
        cur_fiber_tract = all_source_fibers[i]

        warped_fiber_path = '%s/%s_warped.tck' %(str(fiber_out_path), Path(cur_fiber_tract).stem)

        # Deform the source fiber tract
        transformation_utils.warp_fiber_tract(cur_fiber_tract, out_inverse_warp_filename, warped_fiber_path)

        # Compute fiber tract distance
        cur_distance_log_path = '%s/%s_distance.txt' %(str(fiber_distance_path), Path(cur_fiber_tract).stem)
        cur_distance_log = open(cur_distance_log_path, 'w')
        cur_fiber_dist = fiber_measures.fiber_tract_distance(warped_fiber_path, all_target_fibers[i], percent_sample_fibers, num_repeats)
        cur_distance_log.write(str(cur_fiber_dist))