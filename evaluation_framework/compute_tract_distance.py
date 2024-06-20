import argparse

from evaluation_lib import fiber_measures


def main(fiber_tract_M_path, fiber_tract_T_path, percent_sample_fibers):
 
  print(fiber_measures.fiber_tract_distance(fiber_tract_M_path, fiber_tract_T_path, percent_sample_fibers))
  
if __name__=="__main__":

  parser = argparse.ArgumentParser(description='Measures the distance between two fiber tracts in tck format (see Zhang 2006 MEDIA eqn 10)')

  parser.add_argument('--percent_sample_fibers', default=0.5, type=float, help='randomly sample a percentage of fiber streamlines')

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