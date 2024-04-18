import argparse
import vtk
import numpy as np

import fiber_measures
import fiber_tract_io

def main(fiber_tract_M_path, fiber_tract_T_path, percent_sample_fibers):
 
  # Read tck fiber bundles
  fiber_tract_M_header, fiber_tract_M_points_tck, fiber_tract_M_line_starts, fiber_tract_M_line_ends = fiber_tract_io.read_tck(fiber_tract_M_path)
  fiber_tract_T_header, fiber_tract_T_points_tck, fiber_tract_T_line_starts, fiber_tract_T_line_ends = fiber_tract_io.read_tck(fiber_tract_T_path)

  # Fiber bundle M as polydata
  fiber_bundle_M = fiber_tract_io.get_vtk_polydata(fiber_tract_M_points_tck, fiber_tract_M_line_starts, fiber_tract_M_line_ends)

  # Fiber bundle T as polydata
  fiber_bundle_T = fiber_tract_io.get_vtk_polydata(fiber_tract_T_points_tck, fiber_tract_T_line_starts, fiber_tract_T_line_ends)

  # These are all the points which make up the fibers (streamlines) of fiber bundle M
  fiber_tract_M_points = fiber_bundle_M.GetPoints()
  fiber_bundle_M.GetLines().InitTraversal()
  idListM = vtk.vtkIdList()

  # These are all the points which make up fiber bundle T
  fiber_tract_T_points = fiber_bundle_T.GetPoints()

  iter_counter_M = -1  
  index_counter_M = 0

  # Figure out how many fibers in M and T we will use based on the use_every_n_fiber value (needed for the normalization term)
  num_fibers_M = int(np.ceil(fiber_bundle_M.GetNumberOfLines() * percent_sample_fibers))
  num_fibers_T = int(np.ceil(fiber_bundle_T.GetNumberOfLines() * percent_sample_fibers))

  np.random.seed(0)
  range_fibers_M = np.arange(0, fiber_bundle_M.GetNumberOfLines())
  sampling_fibers_M = np.random.choice(range_fibers_M, size=num_fibers_M, replace=False)

  range_fibers_T = np.arange(0, fiber_bundle_T.GetNumberOfLines())
  sampling_fibers_T = np.random.choice(range_fibers_T, size=num_fibers_T, replace=False)

  # This is a matrix which stores all pairwise distances between fibers in M and T -- non-symmetric since d(fm, ft) != d(ft, fm)
  # We cannot store the d in a shared matrix because d() is not symmetric so the diagonal doesn't work in this case
  pairwise_distances_M_and_T = np.zeros((num_fibers_M, num_fibers_T))
  pairwise_distances_T_and_M = np.zeros((num_fibers_T, num_fibers_M))

  # Loop over the fiber bundle M
  while (fiber_bundle_M.GetLines().GetNextCell(idListM)):
   
    iter_counter_M+=1

    if (iter_counter_M not in sampling_fibers_M):
      continue
    
    num_points_fm = idListM.GetNumberOfIds()
    fiber_fm_points = np.zeros((num_points_fm, 3))
    point_counter_fm = 0

    # Go through all points of fiber fm
    for pointId_fm_i in range(idListM.GetNumberOfIds()):
      
      fiber_fm_points[point_counter_fm, :] = fiber_tract_M_points.GetPoint(idListM.GetId(pointId_fm_i))
      point_counter_fm+=1

    fiber_bundle_T.GetLines().InitTraversal()
    idListT = vtk.vtkIdList()   

    iter_counter_T = -1
    index_counter_T = 0

    # Loop over the fiber bundle T
    while (fiber_bundle_T.GetLines().GetNextCell(idListT)):

      iter_counter_T+=1

      if (iter_counter_T not in sampling_fibers_T):
        continue

      num_points_ft = idListT.GetNumberOfIds()
      
      fiber_ft_points = np.zeros((num_points_ft, 3))
      point_counter_ft = 0
      
      # Go through all points of fiber ft
      for pointId_ft_j in range(idListT.GetNumberOfIds()):
          
        fiber_ft_points[point_counter_ft, :] = fiber_tract_T_points.GetPoint(idListT.GetId(pointId_ft_j))
        point_counter_ft+=1

      # Now compute d(fm, ft) and d(ft, fm) store it
      pairwise_distances_M_and_T[index_counter_M, index_counter_T] = fiber_measures.mean_closest_fiber_distance(fiber_fm_points, fiber_ft_points)
      pairwise_distances_T_and_M[index_counter_T, index_counter_M] = fiber_measures.mean_closest_fiber_distance(fiber_ft_points, fiber_fm_points)

      index_counter_T+=1

    index_counter_M+=1

  first_part = np.sum(np.min(pairwise_distances_M_and_T, axis=1))
  second_part = np.sum(np.min(pairwise_distances_T_and_M, axis=1))
  final_value = (1.0 / (num_fibers_M + num_fibers_T)) * (first_part + second_part)
  
  print(final_value)
  
if __name__=="__main__":

  #t = time.process_time()

  parser = argparse.ArgumentParser(description='Measures the distance between two fiber tracts in tck format (see Zhang 2006 MEDIA eqn 10)')

  parser.add_argument('--percent_sample_fibers', default=0.5, type=float, help='Randomly sample a percentage of fiber streamlines')

  parser.add_argument('tract1', type=str, help='path to a tck fiber bundle')
  parser.add_argument('tract2', type=str, help='path to a tck fiber bundle')

  args = parser.parse_args()

  fiber_tract_M_path = args.tract1
  fiber_tract_T_path = args.tract2
  percent_sample_fibers = args.percent_sample_fibers

  # Avoid bad inputs
  if (percent_sample_fibers <= 0.0) or (percent_sample_fibers > 1.0):
    raise Exception("--percent_sample_fibers must be in the range (0.0, 1.0]")
  
  main(fiber_tract_M_path, fiber_tract_T_path, percent_sample_fibers) 