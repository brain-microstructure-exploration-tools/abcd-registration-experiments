import argparse, sys, csv
from doctest import OutputChecker
from pathlib import Path
import numpy as np
import monai

# Where to save scores table
TABLE_NAME = 'fa_reg_scores.csv'

# Directory in which to save aligned images
IMG_SAVE_DIR = 'fa_registered_images'

# Subdirectory of IMG_SAVE_DIR in which to save images that were aligned to a target image with given filename
subdir_from_target_name = lambda target_name : f"aligned_to_{target_name}"

parser = argparse.ArgumentParser(
  description='Nonlinearly register a dataset of affine-aligned fractional anisotropy images into a common space. '+\
    'Images can be in nifti format. Images are cropped or padded to spatial dimensions of (144,144,144) for '+\
    'inference of a deformation field, but any transformations or transformed images that are produced are '+\
    'converted back into the input image coordinates before being saved. All images in the dataset should have the '+\
    'same spatial dimensions for the alignment to make sense. There are two modes for this program, depending on whether '+\
    'the --target argument is provided.'
)

parser.add_argument('dataset',
  help='Path to directory containing dataset.'
)
parser.add_argument('--target', default=None,
  help='Path to target image, which should also be an FA image. If this is provided, then all N images are '+\
    'registered to the target image and there are N registrations to compute. If this is not provided, then '+\
    'all *pairs* of images are registered, for a total of N*(N-1) registrations. Each image is scored based on '+\
    'the mean squared-displacement needed to align all the other images to it. The image with the best score (the smallest '+\
    f'displacement) is pointed out, and a report of all the scores is saved to {TABLE_NAME}.'
)
parser.add_argument('--save-transforms', action='store_true',
  help='(NOT YET IMPLEMENTED) Whether to save the transforms from images to target. The target is either the one given via --target or it '+\
    'is the best scoring image from the evaluation of all pairwise alignments.'
)
parser.add_argument('--save-transformed-images', action='store_true',
  help='Whether to save the transformed images. The target is either the one given via --target or it '+\
    'is the best scoring image from the evaluation of all pairwise alignments.'
)

class ImageLoader:

  def __init__(self, image_only=False):
    self.monai_loader = monai.transforms.LoadImage(image_only=image_only)
    self.post_load_steps = monai.transforms.Compose([
      monai.transforms.AddChannel(),
      monai.transforms.ToTensor(),
    ])
    self.image_only = image_only
  def __call__(self, img_path):

    if self.image_only:
      img = self.monai_loader(img_path)
    else:
      img, header_dict = self.monai_loader(img_path)

    img = self.post_load_steps(img)

    if self.image_only:
      return img
    else:
      return img, header_dict


def reg_all_to_target(image_loader, target_path, fa_paths, reg_model, print_progress=False, save_images=None):
  num_imgs = len(fa_paths)

  save_subdir = None
  if save_images is not None:
    save_subdir = save_images/subdir_from_target_name(target_path.name)
    save_subdir.mkdir(exist_ok=True)
    writer = monai.data.NibabelWriter()

  target_img, _ = image_loader(target_path)
  mean_sqdisplacements = []
  for i,path in enumerate(fa_paths):
    moving_img, moving_img_header = image_loader(path)
    if save_images is None:
      ddf = reg_model.forward(target_img, moving_img)
    else:
      ddf, warped_img = reg_model.forward(target_img, moving_img, include_warped_image=True)
      writer.set_data_array(warped_img, channel_dim=0)
      writer.set_metadata(moving_img_header)
      writer.write(save_subdir/path.name, verbose=False)
      # TODO save warped img to save_subdir/path.name
    mean_sqdisplacement = (ddf**2).sum(axis=0).mean().item()
    mean_sqdisplacements.append(mean_sqdisplacement)
    if print_progress:
      print(f"Registered {i+1} of {num_imgs} images.")
  score = np.mean(mean_sqdisplacements)
  return score

def main(args):
  from fa_deformable_registration_models import reg_model1

  reg_model = reg_model1.RegModel(device='cpu')
  print(f"Registration model loaded to device {reg_model.device}.")

  fa_dir = Path(args.dataset)
  if not fa_dir.is_dir():
    print(f"Invalid dataset directory: {fa_dir}", file=sys.stderr)
    return
  fa_paths = list(fa_dir.glob('*'))
  num_imgs = len(fa_paths)

  if args.save_transforms:
    print("Saving transforms is not yet implemented.", file=sys.stderr)
    return

  saved_img_dir = None
  if args.save_transformed_images:
    saved_img_dir = Path(IMG_SAVE_DIR)
    saved_img_dir.mkdir(exist_ok=True)

  image_loader = ImageLoader(image_only=False)

  if args.target is not None:
    target_path = Path(args.target)
    if not target_path.exists():
      print(f"Target image not found: {target_path}", file=sys.stderr)
      return

    score = reg_all_to_target(image_loader, target_path, fa_paths, reg_model, print_progress=True, save_images=saved_img_dir)
    print(f"Target image score: {score}")
    print("(This is the mean squared displacement needed to align all other images to it.)")
  else:
    names_scores = []
    for i,target_path in enumerate(fa_paths):
      name = target_path.name
      score = reg_all_to_target(image_loader, target_path, fa_paths, reg_model, save_images=saved_img_dir)
      names_scores.append((name,score))
      print(f"Scored {i+1} of {num_imgs} images.")
    with open(TABLE_NAME, 'w', newline='') as csvfile:
      csvwriter = csv.writer(csvfile)
      csvwriter.writerow(['name', 'score'])
      for name, score in names_scores:
        csvwriter.writerow([name, score])
    print(f"Saved table of scores to {TABLE_NAME}.")
    print("(Scores are the mean squared displacement needed to align all other images to a given image.)")
    best_image_name, best_score = min(names_scores, key=lambda p:p[1])
    print(f'The best image is {best_image_name} with a score of {best_score}.')




if __name__=="__main__":
  main(parser.parse_args())
