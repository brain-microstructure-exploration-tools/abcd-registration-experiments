from pathlib import Path
import os
import glob

import ants
import nibabel as nib

import transformation_utils
import fiber_measures

def pairwise_evaluation_ants(target_fa_path, forward_diffeo_path, inverse_diffeo_path, source_fiber_path, target_fiber_path, output_path, percent_sample_fibers=0.1, num_repeats=1):

    ### Convert the transformations ###

    mrtrix_forward_warp = transformation_utils.convert_ants_transform_to_mrtrix_transform(target_fa_path, forward_diffeo_path)
    mrtrix_inverse_warp = transformation_utils.convert_ants_transform_to_mrtrix_transform(target_fa_path, inverse_diffeo_path)

    out_forward_warp_filename = '%s/mrtrix_forward_diffeo.nii.gz' %(str(output_path))
    out_inverse_warp_filename = '%s/mrtrix_inverse_diffeo.nii.gz' %(str(output_path))

    nib.save(mrtrix_forward_warp, out_forward_warp_filename)
    nib.save(mrtrix_inverse_warp, out_inverse_warp_filename)

    ### Measure fiber distance ### (This is a target for refactoring to a seperate module as we add additional scores, all scores could be organized in a csv or similar)

    all_source_fibers = sorted(glob.glob(str(source_fiber_path) + '/*.tck'))
    all_target_fibers = sorted(glob.glob(str(target_fiber_path) + '/*.tck'))

    fiber_out_path = Path('%s/warped_fibers' %(str(output_path)))
    fiber_distance_path = Path('%s/fiber_distances' %(str(output_path)))

    if not fiber_out_path.exists():
        os.mkdir(str(fiber_out_path))

    if not fiber_distance_path.exists():
        os.mkdir(str(fiber_distance_path))

    for i in range(0, len(all_source_fibers)):
 
        cur_fiber_tract = all_source_fibers[i]

        warped_fiber_path = '%s/%s_warped.tck' %(str(fiber_out_path), Path(cur_fiber_tract).stem)

        transformation_utils.warp_fiber_tract(cur_fiber_tract, out_inverse_warp_filename, warped_fiber_path)

        cur_distance_log_path = '%s/%s_distance.txt' %(str(fiber_distance_path), Path(cur_fiber_tract).stem)
        cur_distance_log = open(cur_distance_log_path, 'w')
        cur_fiber_dist = fiber_measures.fiber_tract_distance(warped_fiber_path, all_target_fibers[i], percent_sample_fibers)
        cur_distance_log.write(str(cur_fiber_dist))