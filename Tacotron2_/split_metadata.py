# Training testing split
import argparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metadata', required=True, help='Full path to metadata')
    parser.add_argument('--add_eos', type=bool, default=True, help='Add EOS in sentence')
    parser.add_argument('--train_test_split',type=float, default=0.9, help='train test split data')
    argparse = parser.parse_args()
    addeos = bool(argparse.add_eos)
    path_wavs = argparse.metadata.replace('metadata.csv', '') + 'wavs/'

    with open(argparse.metadata, 'r') as f:
        datas = f.readlines()

    train_ratio = float(argparse.train_test_split)
    i = 0
    train_index = int(train_ratio * len(datas))
    with open('filelists/train.txt', 'w') as fd:
        while i < train_index:
            line_list = datas[i].strip().split("|")
            text, tran = line_list[1], line_list[2]
            if addeos == True:
                text, tran = text + ' .', tran + ' .'
                text, tran = text.replace('. .', '.').replace('? .', '.').replace('! .','.'), \
                             tran.replace('. .', '.').replace('? .', '.').replace('! .','.')
            line = path_wavs + line_list[0] + '.wav' + '|' + text + '|' + tran + '\n'
            fd.write(line)
            i+=1

    with open('filelists/test.txt', 'w') as fd:
        while i < len(datas):
            line_list = datas[i].strip().split("|")
            text, tran = line_list[1], line_list[2]
            if addeos == True:
                text, tran = text + ' .', tran + ' .'
                text, tran = text.replace('. .', '.').replace('? .', '.').replace('! .', '.'), \
                             tran.replace('. .', '.').replace('? .', '.').replace('! .', '.')
            line = path_wavs + line_list[0] + '.wav' + '|' + text + '|' + tran + '\n'
            fd.write(line)
            i += 1