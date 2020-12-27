import numpy as np
import os
# import audio
import torch
from scipy.io.wavfile import read
import audio as Audio
# from tqdm import tqdm
# from functools import partial
# from concurrent.futures import ProcessPoolExecutor


def build_from_path(in_dir, out_dir):
    index = 1
    # executor = ProcessPoolExecutor(max_workers=4)
    # futures = []
    texts = []

    with open(os.path.join(in_dir, 'metadata.csv'), encoding='utf-8') as f:
        for line in f.readlines():
            if index % 100 == 0:
                print("{:d} Done".format(index))
            parts = line.strip().split('|')
            wav_path = os.path.join(in_dir, 'wavs', '%s.wav' % parts[0])
            print(wav_path)
            text = parts[1]
            # futures.append(executor.submit(
            #     partial(_process_utterance, out_dir, index, wav_path, text)))
            texts.append(_process_utterance(out_dir, index, wav_path, text))

            index = index + 1

    # return [future.result() for future in tqdm(futures)]
    return texts


def _process_utterance(out_dir, index, wav_path, text):
    # Compute a mel-scale spectrogram from the wav:
    _, wav = read(wav_path)
    print(wav.shape)
    mel_spectrogram, energy = Audio.tools.get_mel_from_wav(torch.FloatTensor(wav))
    mel_spectrogram = mel_spectrogram.numpy().astype(np.float32)

    # Write the spectrograms to disk:
    mel_filename = 'vlsp2020-mel-%05d.npy' % index
    np.save(os.path.join(out_dir, mel_filename),
            mel_spectrogram.T, allow_pickle=False)

    return text
