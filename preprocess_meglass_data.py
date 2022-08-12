import glob
import os
import random
import shutil
from pathlib import Path

img_path = '/home/phamson/data/sunglasses/MeGlass_120x120'
meta_path = '/home/phamson/data/sunglasses/meta.txt'
normal_glass_path = '/home/phamson/data/sunglasses/normal_glass'

with open(meta_path, 'r') as f:
    lines = [x.strip().split(' ')[0] for x in f.readlines() if x.strip().split(' ')[1] == '1']

glass = []
glob_path = os.path.join(img_path, '*.jpg')
for img in glob.glob(glob_path):
    if Path(img).name in lines:
        glass.append(img)

normal_glass = random.sample(glass, 1700)

for i in normal_glass:
    dst = os.path.join(normal_glass_path, Path(i).name)
    shutil.copyfile(i, dst)
