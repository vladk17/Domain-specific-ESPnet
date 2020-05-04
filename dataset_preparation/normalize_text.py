import sys
import os
import argparse
import shutil
import re
from num2words import num2words


def is_number(s):
    try:
        float(s)
        if s == "infinity" or s == "inf":
            return False
        return True
    except ValueError:
        pass
    return False


specialwords = ['<NUM>']


def toLower(x):
    return " ".join([a if a in specialwords else a.lower() for a in x.split()])


def digit2words(x, L):
    result = " ".join(
        num2words(int(a), lang=L) if a.isdigit() else (num2words(float(a), lang=L) if is_number(a) else a)
        for a in re.split('[ :]', x))
    return result


normalizeEszett = lambda x: " ".join(word[:-1].replace('ß', 'ss') + word[-1]
                                     for word in x.split())

normalizeOK = lambda x: " ".join(a if a != 'ok' else 'okay'
                                 for a in x.split())


def removeExtraSpaces(rest):
    prev = rest
    rest = prev.replace('  ', ' ')
    while (prev != rest):
        prev = rest
        rest = prev.replace('  ', ' ')
    return rest


def normalize(inFile, format, lang):
    infile = open(inFile, 'r').readlines()
    outfile = open(inFile + '_norm', 'w')
    print("Normalizing text to %s" % (inFile + '_norm'))

    for line in infile:
        if line == "\n" or line == "":
            continue
        if format == "text-only":
            name = ""
            rest = ' '.join(line.split())
        else:
            name = line.split()[0]
            rest = ' '.join(line.split()[1:])

        rest = run_normalization(rest, lang)

        if rest == "":
            rest = '<unk>'

        line = ' '.join([name, rest])

        if format == "non-acoustic":
            if len(rest) < 400:
                outfile.write(line + '\n')
            else:
                prev = 0
                for i in range(int(len(rest) / 400) + 1):
                    subStr = rest[prev:min(prev + 400, len(rest))]
                    if prev + 400 < len(rest):
                        words = subStr.split(' ')[:-1]
                    else:
                        words = subStr.split(' ')
                    subStr = ' '.join(words) + '\n'
                    if subStr == "" or subStr == "\n":
                        continue
                    prev = prev + len(subStr)
                    if format == "text-only":
                        outfile.write(subStr)
                    else:
                        outfile.write(' '.join([name + str(prev), subStr]))
        else:
            outfile.write(line + '\n')

    print("DONE")


def run_normalization(rest, lang):
    rest = toLower(rest)
    rest = digit2words(rest, lang)
    rest = remove_non_alphabet_chars(rest, language=lang)
    rest = removeExtraSpaces(rest)
    rest = normalizeEszett(rest)
    rest = normalizeOK(rest)
    rest = rest.replace(' e mail ', ' email ')

    return rest


def remove_non_alphabet_chars(text: str, language: str):
    if language == 'de':
        text = re.sub("[^a-zäöüß]+", " ", text)
    elif language == 'en':
        text = re.sub("[^a-z]+", " ", text)
    elif language == 'es':
        text = re.sub("[^a-záéíóúüñ]+", " ", text)
    else:
        raise RuntimeError("Language is not supported")
    return text


def backup(file):
    dir, name = os.path.split(file)
    bkp = dir + '/.' + name + '.bkp'
    print("Backing up %s to %s" % (file, bkp))
    shutil.copy(file, bkp)


########################
def ParseArgs():
    parser = argparse.ArgumentParser(description="Normalize text file "
                                                 "Usage: normalizeText.py --format --lang <in_text> "
                                                 "E.g. python3.6 normalizeText.py --format acoustic --lang de kaldi_dir/text.orig ",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    parser.add_argument("in_txt_file", help="Input text file")
    parser.add_argument("--format", help="format: non-acoustic , acoustic(default), text-only", default='acoustic')
    parser.add_argument("--lang",
                        help="lang: en (English, default), fr(French), de(German), es(Spanish), lt(Lithuanian), lv(Latvian), en_GB(British English), en_IN(Indian English)",
                        default='en')
    print(' '.join(sys.argv))
    args = parser.parse_args()

    return args


#######################################################################################
if __name__ == '__main__':
    args = ParseArgs()
    print(args)
    _inFile = args.in_txt_file
    _lang = args.lang
    _format = args.format
    backup(_inFile)
    normalize(_inFile, _format, _lang)
