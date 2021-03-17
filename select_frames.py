#!/bin/python
# Randomly select MFCC frames

import argparse
import numpy
import os
import sys
import random
import pickle
from tqdm import tqdm

parser = argparse.ArgumentParser()
parser.add_argument("file_list")
parser.add_argument("surf_path")
parser.add_argument("select_ratio", type=float)
parser.add_argument("output_file")

if __name__ == "__main__":
  args = parser.parse_args()

  fread = open(args.file_list, "r")
  fwrite = open(args.output_file, "w")

  # random selection is done by randomizing the rows of the whole matrix, and then selecting the first
  # num_of_frame * ratio rows
  numpy.random.seed(18877)

  for line in tqdm(fread.readlines()[1:]):  # skipped the header
    surf_path = os.path.join(args.surf_path, line.strip().split(",")[0] + ".p")
    if not os.path.exists(surf_path):
      continue

    featurelist = pickle.load(open(surf_path, 'rb'))
    random.shuffle(featurelist)

    num_frames = len(featurelist)
    if num_frames == 0 or (featurelist[0] is None):
      continue

    feat_dim = featurelist[0].shape[1]

    for frame_i in range(num_frames):
      if featurelist[frame_i] is None:
        continue
      select_size = int(len(featurelist[frame_i]) * args.select_ratio)

      for feature in range(select_size):
        line = str(featurelist[frame_i][feature][0])
        for m in range(1, feat_dim):
          line += ";" + str(featurelist[frame_i][feature][m])
        fwrite.write(line + "\n")


  fwrite.close()
