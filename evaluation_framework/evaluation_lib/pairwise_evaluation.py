from pathlib import Path
import os
import glob

import ants
import nibabel as nib

from evaluation_lib import transformation_utils
from evaluation_lib import fiber_measures

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


def pairwise_evaluation_ants(
    target_fa_path: Path, forward_diffeo_path: Path, inverse_diffeo_path: Path, 
    source_fiber_path: Path, target_fiber_path: Path, 
    output_path: Path, percent_sample_fibers: float=0.1, num_repeats: int=1, specified_fibers: list=[]
) -> None:
    """
    Performs a pairwise registration evalutation given the output of ants registration and directories for source and target fiber tracts
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
    :param specified_fibers: a list of strings specifying which fibers to use
    """

    ### Convert the transformations to mrtrix format ###

    mrtrix_forward_warp = transformation_utils.convert_ants_transform_to_mrtrix_transform(target_fa_path, forward_diffeo_path)
    mrtrix_inverse_warp = transformation_utils.convert_ants_transform_to_mrtrix_transform(target_fa_path, inverse_diffeo_path)

    out_forward_warp_filename = '%s/mrtrix_forward_diffeo.nii.gz' %(str(output_path))
    out_inverse_warp_filename = '%s/mrtrix_inverse_diffeo.nii.gz' %(str(output_path))

    nib.save(mrtrix_forward_warp, out_forward_warp_filename)
    nib.save(mrtrix_inverse_warp, out_inverse_warp_filename)

    ### Measure fiber distance ### (This is a target for refactoring to a seperate module as we add additional scores, all scores could be organized in a csv or similar)

    # If the user did not specify which subset of fibers to use, we will use the "defaul set" (which we can determine later -- right now its all tracts)
    if (len(specified_fibers) == 0):

        specified_fibers = DEFAULT_FIBER_TRACTS    

    # Output folders for warped fibers and for fiber tract distances
    fiber_out_path = Path('%s/warped_fibers' %(str(output_path)))

    if not fiber_out_path.exists():
        os.mkdir(str(fiber_out_path))

    fiber_distance_csv_path = output_path / 'fiber_distances.csv'
    fiber_distance_csv = ""

    for i in range(0, len(specified_fibers)):

        cur_fiber_tract = source_fiber_path / specified_fibers[i]
        target_fiber_tract = target_fiber_path / specified_fibers[i]

        # If it's not there, skip it
        if (not cur_fiber_tract.exists() or not target_fiber_tract.exists()):
            continue

        warped_fiber_path = '%s/%s_warped.tck' %(str(fiber_out_path), Path(cur_fiber_tract).stem)

        # Deform the source fiber tract
        transformation_utils.warp_fiber_tract(cur_fiber_tract, out_inverse_warp_filename, warped_fiber_path)

        # Compute fiber tract distance
        cur_fiber_dist = fiber_measures.fiber_tract_distance(warped_fiber_path, target_fiber_tract, percent_sample_fibers, num_repeats)
        fiber_distance_csv += specified_fibers[i] + ',' + str(cur_fiber_dist) + '\n'

    # Write fiber distance csv
    f = open(str(fiber_distance_csv_path), 'w')
    f.write(fiber_distance_csv)
    f.close()