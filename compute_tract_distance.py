import argparse
import vtk
import numpy as np

# This will be the mean closest fiber distance
# Gerig et al. (2004) as ‘‘the mean of closest distances’’
# It is defined as ‘‘the mean of the closest distance for every point of two fibers’’
# 
# Go through all points of fiber fm and find the point on fiber ft that is closest
# Measure this distance and record it
# Compute the average of all these distances
def fiber_distance(points_m, points_t):
  
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

def main(fiber_bundle_M_path, fiber_bundle_T_path, use_every_n_fibers):

  # Read fiber bundle M as polydata
  reader_M = vtk.vtkPolyDataReader()
  reader_M.SetFileName(fiber_bundle_M_path)
  reader_M.Update()
  fiber_bundle_M = reader_M.GetOutput()

  # Read fiber bundle T as polydata
  reader_T = vtk.vtkPolyDataReader()
  reader_T.SetFileName(fiber_bundle_T_path)
  reader_T.Update()
  fiber_bundle_T = reader_T.GetOutput()

  # These are all the points which make up the fibers (streamlines) of fiber bundle M
  fiber_bundle_M_points = fiber_bundle_M.GetPoints()
  fiber_bundle_M.GetLines().InitTraversal()
  idListM = vtk.vtkIdList()

  # These are all the points which make up fiber bundle T
  fiber_bundle_T_points = fiber_bundle_T.GetPoints()

  iter_counter_M = -1  
  index_counter_M = 0

  # Figure out how many fibers in M and T we will use based on the use_every_n_fiber value (needed for the normalization term)
  num_fibers_M = int(np.ceil(fiber_bundle_M.GetNumberOfLines()/use_every_n_fibers))
  num_fibers_T = int(np.ceil(fiber_bundle_T.GetNumberOfLines()/use_every_n_fibers))

  # This is a matrix which stores all pairwise distances between fibers in M and T -- non-symmetric since d(fm, ft) != d(ft, fm)
  # We cannot store the d in a shared matrix because d() is not symmetric so the diagonal doesn't work in this case
  pairwise_distances_M_and_T = np.zeros((num_fibers_M, num_fibers_T))
  pairwise_distances_T_and_M = np.zeros((num_fibers_T, num_fibers_M))

  # Loop over the fiber bundle M
  while (fiber_bundle_M.GetLines().GetNextCell(idListM)):
   
    iter_counter_M+=1

    if (iter_counter_M % use_every_n_fibers != 0):
      continue
    
    fiber_fm_points = np.zeros((idListM.GetNumberOfIds(), 3))
    point_counter_fm = 0
    
    # Go through all points of fiber fm
    for pointId_fm_i in range(idListM.GetNumberOfIds()):

      fiber_fm_points[point_counter_fm, :] = fiber_bundle_M_points.GetPoint(pointId_fm_i)
      point_counter_fm+=1
      
    fiber_bundle_T.GetLines().InitTraversal()
    idListT = vtk.vtkIdList()   

    iter_counter_T = -1
    index_counter_T = 0

    # Loop over the fiber bundle T
    while (fiber_bundle_T.GetLines().GetNextCell(idListT)):

      iter_counter_T+=1

      if (iter_counter_T % use_every_n_fibers != 0):
        continue

      fiber_ft_points = np.zeros((idListT.GetNumberOfIds(), 3))
      point_counter_ft = 0
      
      # Go through all points of fiber ft
      for pointId_ft_j in range(idListT.GetNumberOfIds()):
        
        fiber_ft_points[point_counter_ft, :] = fiber_bundle_T_points.GetPoint(pointId_ft_j)
        point_counter_ft+=1
        
      # Now compute d(fm, ft) and d(ft, fm) store it
      pairwise_distances_M_and_T[index_counter_M, index_counter_T] = fiber_distance(fiber_fm_points, fiber_ft_points)
      pairwise_distances_T_and_M[index_counter_T, index_counter_M] = fiber_distance(fiber_ft_points, fiber_fm_points)

      index_counter_T+=1

    index_counter_M+=1

  first_part = np.sum(np.min(pairwise_distances_M_and_T, axis=1))
  second_part = np.sum(np.min(pairwise_distances_T_and_M, axis=1))
  final_value = (1.0 / (num_fibers_M + num_fibers_T)) * (first_part + second_part)
  
  print()
  print(final_value)
  print()

if __name__=="__main__":

  parser = argparse.ArgumentParser(description='Measures the distance between two fiber bundles (see Zhang 2006 MEDIA eqn 10)')

  parser.add_argument('--use_every_n', default=10, type=int, help='Sample every n fiber streamlines rather than all fibers')

  parser.add_argument('bundle1', type=str, help='path to a fiber bundle')
  parser.add_argument('bundle2', type=str, help='path to a fiber bundle')

  args = parser.parse_args()

  fiber_bundle_M_path = args.bundle1
  fiber_bundle_T_path = args.bundle2
  use_every_n_fibers = args.use_every_n

  # Prevent a divide by zero error later and avoid negative values 
  if (use_every_n_fibers <= 0):
    use_every_n_fibers = 1

  main(fiber_bundle_M_path, fiber_bundle_T_path, use_every_n_fibers) 
