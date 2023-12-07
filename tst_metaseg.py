from metaseg import SegAutoMaskPredictor, SegManualMaskPredictor

# If gpu memory is not enough, reduce the points_per_side and points_per_batch.

# For image
results = SegAutoMaskPredictor().image_predict(
    source="./seg_tst/000126.png",
    model_type="vit_l", # vit_l, vit_h, vit_b
    points_per_side=16, 
    points_per_batch=64,
    min_area=0,
    output_path="output.jpg",
    show=False,
    save=False,
)

import pdb; pdb.set_trace()

