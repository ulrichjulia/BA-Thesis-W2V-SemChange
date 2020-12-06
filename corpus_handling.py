import os, nltk, re, logging
from pathlib import Path
from langdetect import detect, lang_detect_exception
from main_project import compile_era_list as comp
from nltk.stem import WordNetLemmatizer

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

wnl = WordNetLemmatizer()


def replace_inflections():
    """
    For all the words to be analyzed, the reflected forms (e.g., 'friends', 'liking' etc.)
        are replaced with the uninflected form as it is on the list.
        """
    cupath = str(Path.cwd())
    newpath = cupath+'/Data/Offline/'
    dirs = os.listdir(newpath)

    for dir in dirs:
        if dir.endswith('Store') != True:
            nedir = os.listdir(newpath+dir)
            for ned in nedir:
                if ned.endswith('Store') != True:
                    new = os.listdir(newpath+dir+'/'+ned)
                    for file in new:
                        if file.endswith('.txt'):
                            filepa = newpath+dir+'/'+ned+'/'+file
                            with open(filepa, 'r') as f:
                                text = f.readlines()
                            with open(filepa, 'w') as f:
                                for line in text:
                                    line = re.sub('friend\w+', 'friend', line)
                                    line = re.sub('ship\w+', 'ship', line)
                                    line = re.sub('invit\w+', 'invite', line)
                                    line = re.sub('follow\w+', 'follow', line)
                                    line = re.sub('lik\w+', 'like', line)
                                    line = re.sub('stan\w+', 'stan', line)
                                    f.write(line)


def count_tokens_in_subcorpus(folder: str, mode: str, descr: str):
    """
    For the specified date (folder) and mode ('i' for internet, 'o' for offline),
        all of the tokens in the subcorpus are counted and written into a file.
        """
    eras = comp(folder, mode)
    totaltokens = []

    for dir in eras:
        dirs = os.listdir(dir)
        for file in dirs:
            if file.endswith('.txt'):
                file_path = dir + '/' + file
                for line in open(file_path).readlines():
                    if line.startswith('# ') != True:
                        sent = nltk.tokenize.word_tokenize(line)
                        for item in sent:
                            if item.isalpha():
                                totaltokens.append(item)

    with open('Token Count.txt', 'a') as resfi:
        resfi.write('\n'+'The total numbers of token for the {} documents in {} is '.format(descr, folder)+str(len(totaltokens))+'.\n')


def overall_tokens(file: str):
    """
    After the tokens have been counted for each subcorpus, the overall token count for the whole dataset is calculated.
        """
    fupath = str(Path.cwd()) + '/' + file
    datelist = '1850, 1900, 1930, 1961, 1991, 1992, 2004, 2005, 2006, 2012, 2013, 2017, 2020'
    numberlist = []

    with open(fupath, 'r') as f:
        lines = f.read()
        for numb in re.findall(r'\d+', lines):
            if not re.search(str(numb), datelist):
                numberlist.append(int(numb))
    sumto = sum(numberlist)
    with open(fupath, 'a') as f:
        f.write('\nThe total number of tokens used for the paper is: {}!'.format(str(sumto)))



def prepro_redtype(folder: str):
    """
    Specific function for preprocessing the text files scraped from Reddit and TypePad.
        """
    fupath = str(Path.cwd()) + '/Data/Internet/Others/' + folder
    files = os.listdir(fupath)
    forbidto = ['reddit', 'Reddit', 'typepad', 'TypePad', 'FEED', 'hottestnewesttop', 'all-timestats', 'Register', '»', 'Password']

    for file in files:
        if file.endswith('.txt'):
            with open(fupath+'/'+file, 'r') as f:
                text = f.readlines()

            with open(fupath+'/'+file, 'w') as f:
                for line in text:
                    try:
                        if not re.match('this data is currently not publicly accessible', line) and not re.match('want to join\?', line) and not re.match('login/registerusername:password', line) and not re.match('[\w-]+\s\(\d+\)', line):
                            notre = []
                            tokens = nltk.tokenize.word_tokenize(line)
                            for token in tokens:
                                if token not in forbidto and len(token) < 15 and token.isalpha() and detect(token) == 'en':
                                    notre.append(token)
                            try:
                                if not len(notre)/len(tokens) <= 0.25 and len(tokens) > 1:
                                    f.write(line)
                                else:
                                    continue
                            except ZeroDivisionError:
                                continue
                    except lang_detect_exception.LangDetectException:
                        continue


def det_del_lang(folder: str):
    """
    Those downloaded Instagram posts which are not in English are removed from the data.
        """
    fupath = str(Path.cwd()) + '/Data/Internet/' + folder
    data = os.listdir(fupath)

    for file in data:
        if file.endswith('.txt'):
            with open(fupath+'/'+file, 'r') as f:
                text = f.readlines()
                detli = []
                try:
                    for line in text:
                        if detect(line) == 'en':
                            detli.append(0)
                        elif detect(line) != 'en':
                            detli.append(1)

                    if sum(detli)/len(detli) <= 0.65:
                        continue
                    else:
                        os.remove(fupath+'/'+file)
                except Exception:
                    continue


def delete_spam_posts(folder: str):
    """
    Those downloaded Instagram posts which consist of 40% or more hashtags are removed from the data.
        """
    fupath = str(Path.cwd()) + '/Data/Internet/' + folder
    data = os.listdir(fupath)
    spampatt = r'#[\w\d]*'
    pupa = r'[\.\?\+\\\*\$\^\{\}\[\]\(\)\|!:,;-–—\s\W]*'

    for file in data:
        if file.endswith('.txt'):
            totspam = []
            tottok = []
            with open(fupath+'/'+file, 'r') as f:
                try:
                    for line in f.readlines():
                        se = [word for word in re.findall(spampatt, line)]
                        tok = [token for token in line.split(' ')]
                        for word in se:
                            totspam.append(word)
                        for token in tok:
                            if token != '\n' and token not in pupa:
                                tottok.append(token)
                except UnicodeDecodeError:
                    continue

                if len(totspam)/(len(tottok)+len(totspam)) >= 0.4:
                    os.remove(fupath+'/'+file)
