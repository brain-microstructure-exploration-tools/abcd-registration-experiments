import os
import numpy as np
import vtk

import fiber_tract_io

# This will be the mean closest fiber distance
# Gerig et al. (2004) as ‘‘the mean of closest distances’’
# It is defined as ‘‘the mean of the closest distance for every point of two fibers’’
# 
# Go through all points of fiber fm and find the point on fiber ft that is closest
# Measure this distance and record it
# Compute the average of all these distances
def mean_closest_fiber_distance(points_m, points_t):
  
  # Reshape for vectorized calculation
  points_m = points_m.reshape((len(points_m), 1, 3))
  points_t = points_t.reshape((1, len(points_t), 3))

  # fm - ft (num_points_f, num_points_t, 3) -- for an element i,j this is the coordinate distance between fm_i and ft_j
  fm_minus_ft = points_m - points_t
  squared = fm_minus_ft**2
  
  # We should sum 3rd dim (x,y,z) coords and take the sqrt
  sum_of_squares = np.sum(squared, axis=2)
  distances_fm_ft = np.sqrt(sum_of_squares)

  # Now we the need the minimum value of each row which should represent for point fm_i the closest point on ft
  mins_fm = np.min(distances_fm_ft, axis=1)

  return np.mean(mins_fm)

# Fiber tract distance defined by eqn (3) https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9273049/
def fiber_tract_distance(fiber_tract_M_tck_path, fiber_tract_T_tck_path, percent_sample_fibers=0.5):
  
    # Read tck fiber bundles
    fiber_tract_M_header, fiber_tract_M_points, fiber_tract_M_line_starts, fiber_tract_M_line_ends = fiber_tract_io.read_tck(fiber_tract_M_tck_path)
    fiber_tract_T_header, fiber_tract_T_points, fiber_tract_T_line_starts, fiber_tract_T_line_ends = fiber_tract_io.read_tck(fiber_tract_T_tck_path)

    # Figure out how many fibers in M and T we will use based on the use_every_n_fiber value (needed for the normalization term)
    num_fibers_M = int(np.ceil(len(fiber_tract_M_line_ends) * percent_sample_fibers))
    num_fibers_T = int(np.ceil(len(fiber_tract_T_line_ends) * percent_sample_fibers))

    np.random.seed(0)
    range_fibers_M = np.arange(0, len(fiber_tract_M_line_ends))
    sampling_fibers_M = np.random.choice(range_fibers_M, size=num_fibers_M, replace=False)

    range_fibers_T = np.arange(0, len(fiber_tract_T_line_ends))
    sampling_fibers_T = np.random.choice(range_fibers_T, size=num_fibers_T, replace=False)

    # This is a matrix which stores all pairwise distances between fibers in M and T -- non-symmetric since d(fm, ft) != d(ft, fm)
    # We cannot store the d in a shared matrix because d() is not symmetric so the diagonal doesn't work in this case
    pairwise_distances_M_and_T = np.zeros((num_fibers_M, num_fibers_T))
    pairwise_distances_T_and_M = np.zeros((num_fibers_T, num_fibers_M))

    index_counter_M = 0

    # Loop over the fibers in fiber tract T
    for m in range(0, len(fiber_tract_M_line_ends)):

        if (m not in sampling_fibers_M):
            continue

        fiber_M_point_indices = np.arange(fiber_tract_M_line_starts[m], fiber_tract_M_line_ends[m]+1)
        fiber_fm_points = fiber_tract_M_points[fiber_M_point_indices, :]

        index_counter_T = 0

        for t in range(0, len(fiber_tract_T_line_ends)):

            if (t not in sampling_fibers_T):
                continue

            fiber_T_point_indices = np.arange(fiber_tract_T_line_starts[t], fiber_tract_T_line_ends[t]+1)
            fiber_ft_points = fiber_tract_T_points[fiber_T_point_indices, :]

            pairwise_distances_M_and_T[index_counter_M, index_counter_T] = mean_closest_fiber_distance(fiber_fm_points, fiber_ft_points)
            pairwise_distances_T_and_M[index_counter_T, index_counter_M] = mean_closest_fiber_distance(fiber_ft_points, fiber_fm_points)

            index_counter_T+=1

        index_counter_M+=1

    first_part = np.sum(np.min(pairwise_distances_M_and_T, axis=1))
    second_part = np.sum(np.min(pairwise_distances_T_and_M, axis=1))
    
    return (1.0 / (num_fibers_M + num_fibers_T)) * (first_part + second_part)
