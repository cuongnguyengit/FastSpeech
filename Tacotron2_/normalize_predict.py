import string, re
from vinorm import TTSnorm
from text import text_to_sequence
import torch
import numpy as np

class Normalize:
    def __init__(self, tacotron, waveglow):
        self.model = tacotron
        self.waveglow = waveglow

    def normalize_text(self, sentence):
        print(sentence)
        text = TTSnorm(sentence)
        text = text.lower().translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
        translator = str.maketrans(string.punctuation, ',' * len(string.punctuation))
        text = text.translate(translator)
        text = ' '.join(text.split())
        text = re.sub(r'(\,\ *)+', ',', text)
        text = text.translate(str.maketrans({key: " {0} ".format(key) for key in string.punctuation}))
        text = (text.strip() + ' .').replace(', .', ' .')
        text = re.sub(' +', ' ', text)
        # text = ViTokenizer.tokenize(text)
        # texts = text.split(',')
        return text

    def tts(self, text):
        print("TTS", text)
        text = self.normalize_text(text)
        sequence = np.array(text_to_sequence(text.lower(), ['basic_cleaners']))[None, :]
        sequence = torch.autograd.Variable(
            torch.from_numpy(sequence)).cuda().long()
        mel_outputs, mel_outputs_postnet, _, alignments = self.model.inference(sequence)
        with torch.no_grad():
            audio = self.waveglow.infer(mel_outputs_postnet, sigma=1.0)

        # audio = audio[0].data.cpu().numpy()

        return audio

    def pts(self, para):
        audio = np.zeros(int(0.1 * 22050))
        sentence_ls = para.split(".")
        temp_audio = np.zeros(int(0.45 * 22050))
        temp_sub_audio = np.zeros(int(0.25 * 22050))
        begin = False

        for idx in range(len(sentence_ls)):
            sen = sentence_ls[idx]
            if sen != '' and sen != ' ':
                sub_stn_ls = re.split(",|;|-|:", sen)
                begin_sub = False
                audio_sub = np.zeros(int(0.1 * 22050))
                for idx_sub in range(len(sub_stn_ls)):
                    sub_stn = sub_stn_ls[idx_sub]
                    if sub_stn != '' and sub_stn != ' ':
                        audio_ = self.tts(sub_stn)
                        if begin_sub == False:
                            audio_sub = audio_
                            begin_sub = True
                        else:
                            audio_sub = np.concatenate((audio_sub, temp_sub_audio), axis=0)
                            audio_sub = np.concatenate((audio_sub, audio_), axis=0)
                if begin == False:
                    audio = audio_sub
                    begin = True
                else:
                    audio = np.concatenate((audio, temp_audio), axis=0)
                    audio = np.concatenate((audio, audio_sub), axis=0)
        return audio


