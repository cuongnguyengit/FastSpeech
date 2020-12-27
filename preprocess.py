import argparse
import numpy as np
import shutil
import os

from data import ljspeech
import hparams as hp


def preprocess_ljspeech(filename):
    in_dir = filename
    out_dir = hp.mel_ground_truth
    if not os.path.exists(out_dir):
        os.makedirs(out_dir, exist_ok=True)
    metadata = ljspeech.build_from_path(in_dir, out_dir)
    write_metadata(metadata, out_dir)

    shutil.move(os.path.join(hp.mel_ground_truth, "train.txt"),
                os.path.join("data", "train.txt"))


def write_metadata(metadata, out_dir):
    with open(os.path.join(out_dir, 'train.txt'), 'w', encoding='utf-8') as f:
        for m in metadata:
            f.write(m + '\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, default='./')
    args = parser.parse_args()
    preprocess_ljspeech(args.path)
