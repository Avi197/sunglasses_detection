import os
import glob

no_glasses = '/home/phamson/data/sunglasses/archive/glasses_noGlasses (another copy)/no_glass'
out_no_glasses = '/home/phamson/data/sunglasses/archive/glasses_noGlasses (another copy)/no_glasses'

sunglasses = '/home/phamson/data/sunglasses/archive/glasses_noGlasses (another copy)/with_glasses'
out_sunglasses = '/home/phamson/data/sunglasses/archive/glasses_noGlasses (another copy)/sunglasses'

norm_glasses = '/home/phamson/data/sunglasses/normal_glass'
out_norm_glasses = '/home/phamson/data/sunglasses/yolo_eyeglass/normglasses'
# for idx, file in enumerate(glob.glob(os.path.join(sunglasses, '*'))):
#     os.rename(file, os.path.join(out_sunglasses, f'{idx}.jpg'))

for idx, file in enumerate(glob.glob(os.path.join(norm_glasses, '*'))):
    os.rename(file, os.path.join(out_norm_glasses, f'{idx}.jpg'))
