import os
import sys

import matplotlib.pyplot as plt
import pandas as pd

# Vision
import timm
from competition_utils import setup_competition
from fastai.callback.schedule import slide, valley
from fastai.data.transforms import get_image_files

# Fastai and core
from fastai.metrics import error_rate
from fastai.vision.augment import Resize, aug_transforms
from fastai.vision.core import PILImage
from fastai.vision.data import ImageDataLoaders
from fastai.vision.learner import vision_learner

# Competition and configs
from configs import competition, data_path


def pipeline(debug=True):
    # Setup competition
    path = setup_competition(
        competition, 
        data_path, 
        install='"fastcore>=1.4.5" "fastai>=2.7.1" "timm>=0.6.2.dev0"'
    )

    # Training path
    trn_path = path/'train_images'
    files = get_image_files(trn_path)

    # Training dataloaders
    dls = ImageDataLoaders.from_folder(trn_path, valid_pct=0.2, seed=42,
        item_tfms=Resize(480, method='squish'),
        batch_tfms=aug_transforms(size=128, min_scale=0.75)
    )
    if debug:
        dls.show_batch(max_n=6)
        plt.show()

    # Vision learner
    learn = vision_learner(
        dls, 
        'resnet26d', 
        metrics=error_rate, 
        path='.'
    )
    
    # Train
    learn.fine_tune(3, 0.01)

    # Evaluate
    ss = pd.read_csv(path/'sample_submission.csv')
    tst_files = get_image_files(path/'test_images').sorted()
    tst_dl = dls.test_dl(tst_files)
    probs,_,idxs = learn.get_preds(dl=tst_dl, with_decoded=True)
    mapping = dict(enumerate(dls.vocab))
    results = pd.Series(idxs.numpy(), name="idxs").map(mapping)
    ss['label'] = results
    ss.to_csv(path/'subm.csv', index=False)
    os.system(f"head {path}/subm.csv")


if __name__ == "__main__":
    pipeline()







