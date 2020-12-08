import re, os, time, nltk, logging, random, gensim.models
from pathlib import Path
from math import floor as flo
import numpy as np
from sklearn.manifold import TSNE
import matplotlib.pyplot as matpl
import matplotlib.patches as mpat
from nltk.stem import WordNetLemmatizer

logging.basicConfig(format="%(levelname)s - %(asctime)s: %(message)s", datefmt= '%H:%M:%S', level=logging.INFO)

wnl = WordNetLemmatizer()

words = open(str(Path.cwd())+'/words', 'r', encoding='utf-8').readlines()
words = [re.sub('\n', '', word) for word in words]


def compile_era_list(era: int, mode: str):
    """
    Create a complete list of all files belonging to a specified era in the desired mode ('i' for internet, 'o' for offline).
        """
    if mode == 'o':
        data_folder = str(Path.cwd()) + '/Data/Offline/'
        datadir = os.listdir(data_folder)
    elif mode == 'i':
        data_folder = str(Path.cwd()) + '/Data/Internet/'
        datadir = os.listdir(data_folder)

    eralist = []

    for dir in datadir:
        if os.path.isdir(data_folder+dir):
            indi = os.listdir(data_folder+dir)
            for inn in indi:
                if inn.startswith(str(era)) and not inn.endswith('.txt'):
                    eralist.append(data_folder+dir+'/'+inn)
    return eralist


def read_input(era: list, mode: str):
    """
    The data which is to be processed is read, tokenized returned as a list ready for building the model.
        """
    eras = compile_era_list(era, mode)
    listofresult = []
    for dir in eras:
        dirs = os.listdir(dir)
        for file in dirs:
            if file.endswith('.txt'):
                file_path = dir + '/' + file                                        #TODO: beautify whole code!!!
                with open(file_path, 'r') as f:
                    text = f.read()
                    text = re.sub(r'#.+', '', text)

                    all_sentences = nltk.sent_tokenize(text.lower())

                    all_words = [nltk.word_tokenize(sent) for sent in all_sentences]

                    for item in all_words:
                        listofresult.append([wnl.lemmatize(word) for word in item if word.isalpha() and word.isascii() and 16 > len(word) > 2 and word != 'http' and word != 'https'])
    return listofresult


def train_model(dataname, mode: str):
    """
    The data as prepared by read_input() is used to build a Word2Vec model which subsequently stored.
        The different configurations for varying eras and modes were chosed based on tests for the best results.
        Importantly, the size of the vectors always stays the same (size=128).
        """
    sent = read_input(dataname, mode)

    if dataname == 1850:
        w2v_model = gensim.models.Word2Vec(min_count=5, iter=15, max_final_vocab=10000, size=128, alpha=0.03, min_alpha=0.007, workers=4, seed=5, compute_loss=True)
    elif dataname == 1900:
        w2v_model = gensim.models.Word2Vec(min_count=3, window=10, iter=5, max_final_vocab=1500, size=128, alpha=0.03, min_alpha=0.007, workers=4, seed=5, compute_loss=True)
    elif dataname == 1930:
        w2v_model = gensim.models.Word2Vec(min_count=4, window=6, iter=10, max_final_vocab=20000, size=128, alpha=0.03, sample=0.01, min_alpha=0.007, workers=1, seed=5, compute_loss=True)
    elif dataname == 1961:
        w2v_model = gensim.models.Word2Vec(min_count=3, window=15, iter=10, max_final_vocab=10000, size=128, alpha=0.03, min_alpha=0.007, workers=4, seed=5, compute_loss=True)
    elif dataname == 2020 and mode == 'o':
        w2v_model = gensim.models.Word2Vec(min_count=5, window=6, iter=10, size=128, sample=0.001, alpha=0.03, min_alpha=0.007, workers=4, seed=5, compute_loss=True)
    elif dataname == 2004 and mode == 'i':
        w2v_model = gensim.models.Word2Vec(min_count=3, window=6, iter=7, size=128, sample=0.01, alpha=0.03, min_alpha=0.007, workers=4, seed=5, compute_loss=True)
    elif dataname == 2006 and mode == 'i':
        w2v_model = gensim.models.Word2Vec(min_count=5, window=10, iter=10, max_final_vocab=20000, size=128, sample=0.001, alpha=0.03, min_alpha=0.007, workers=4, seed=5, compute_loss=True)
    elif dataname == 2012 and mode == 'i':
        w2v_model = gensim.models.Word2Vec(min_count=7, window=4, iter=12, max_final_vocab=15000, size=128, alpha=0.03, min_alpha=0.007, workers=4, seed=5, compute_loss=True)
    else:
        w2v_model = gensim.models.Word2Vec(min_count=5, window=8, iter=15, size=128, sample=6e-5, alpha=0.03, min_alpha=0.007, workers=4, seed=5, compute_loss=True)    #the standard configuration

    t = time.time()
    w2v_model.build_vocab(sent, progress_per=10000)
    print('Time to build vocab: {} mins'.format(round((time.time() - t) / 60, 2)))
    t = time.time()
    w2v_model.train(sent, total_examples=w2v_model.corpus_count, epochs=30, report_delay=1)
    print('Time to train the model: {} mins'.format(round((time.time() - t) / 60, 2)))
    w2v_model.init_sims(replace=True)
    w2v_model.save(os.path.join(str(Path.cwd())+'/Models unaligned/'+'W2V_model_{}_{}'.format(dataname, mode)))


def plot_with_matplotlib(path, size: int, att: int):
    """
    The specified model (path=='Models') or models (type(path)==list) are loaded.
        The vectors for the words on the wordlist are created and for the three on which the main-focus lies,
        the most similar 12 words to them are extracted and their respective vectors are calculated.
        Then, the vectors are mathematically downsized to two dimensions using T_SNE.
        Finally, the thus obtained numbers are plotted on a graph in a colour-coordinated manner.
        """
    if path == 'Models aligned':
        data_folder = str(Path.cwd()) + '/' + path
        datadir = os.listdir(data_folder)
    elif type(path) == list:
        data_folder = str(Path.cwd()) + '/Models aligned'
        datadir = []
        for item in path:
            datadir.append(item)

    for dir in datadir:
        if dir.endswith('_o_compl_aligned') or dir.endswith('_i_compl_aligned'):
            fulpa = data_folder + '/' + dir
            model = gensim.models.Word2Vec.load(fulpa)

            vectors = []
            labels = []

            for word in words:
                if words.index(word) > 2:
                    try:
                        vectors.append(model.wv[word])
                        labels.append(word)
                        simiwords = model.wv.most_similar(word, topn=12)
                        for simi, v in simiwords:
                            if simi not in words and simi not in labels:
                                vectors.append(model.wv[simi])
                                labels.append(simi)
                            else:
                                continue
                    except KeyError:
                        continue
                else:
                    try:
                        vectors.append(model.wv[word])
                        labels.append(word)
                    except KeyError:
                        continue

            labels = np.asarray(labels)
            vector = np.asarray(vectors)
            tsne = TSNE(n_components=2, random_state=0)
            vectors = tsne.fit_transform(vector)

            x_vals = [v[0] for v in vectors]
            y_vals = [v[1] for v in vectors]

            nam = re.findall('\d{4}-?\d?\d?', dir)[0]
            if re.search('_o', dir):
                mode = 'Offline'
            elif re.search('_i', dir):
                mode = 'Internet'

            random.seed(0)

            matpl.figure(figsize=(10, 10))
            matpl.xlim(-size, size)
            matpl.ylim(-size, size)
            matpl.xlabel('Reduced Dimension 1', fontsize=18, fontweight='normal', labelpad=30)
            matpl.ylabel('Reduced Dimension 2', fontsize=18, fontweight='normal', va='bottom', ha='center')
            matpl.title('Distribution of Word Embeddings: {} {}'.format(nam, mode), fontsize=20, fontweight='normal', va='bottom')
            matpl.grid()

            i = -3
            colours = ['lightcoral', 'paleturquoise', 'indigo', 'palevioletred', 'peachpuff']
            labels = list(labels)

            for label in labels:
                noum = labels.index(label)
                if label in words[:3]:
                    matpl.scatter(x_vals[noum], y_vals[noum], color='lime')
                    matpl.annotate(str(labels[noum]).lower(), xy=(x_vals[noum], y_vals[noum]), fontsize=14, fontweight='light', ha='right', va='bottom')
                    if len(labels) == 41:
                        i += 3/2
                    else:
                        i += 1
                else:
                    co = flo(i/13)
                    matpl.scatter(x_vals[noum], y_vals[noum], color=colours[co])
                    if label not in words:
                        matpl.annotate(str(labels[noum]).lower(), xy=(x_vals[noum], y_vals[noum]), fontsize=14, fontweight='light', ha='right', va='bottom')
                        i += 1
                    else:
                        matpl.annotate(str(labels[noum]).upper(), xy=(x_vals[noum], y_vals[noum]), fontsize=14, fontweight='light', ha='right', va='bottom')
                        i += 1

            figname = 'figure_{}_{}_{}'.format(nam, mode, att)
            matpl.savefig(os.path.join(str(Path.cwd())+'/Graphs/'+figname))

def plot_with_matplotlib_single(path, mode, size: int, att: int):
    """
    The specified model (path=='Models') or models (type(path)==list) are loaded.
        The vectors for the main-focus words on the wordlist are created for each point in time for which data was collected.
        Then, the vectors are mathematically downsized to two dimensions using T_SNE.
        Finally, the thus obtained vectors are plotted on a graph in in such a way that
        the dot for the odlest dataset is the lightest and the dots become increasingly more saturated.
        The dots are numbered, 1 is the oldest and the dot labelled with the token is the most recent one.
        """
    if path == 'Models aligned':
        data_folder = str(Path.cwd()) + '/' + path
        datadir = os.listdir(data_folder)
    elif type(path) == list:
        data_folder = str(Path.cwd()) + '/Models aligned'
        datadir = []
        for item in path:
            datadir.append(item)

    vectorsfr = []
    labelsfr = []
    vectorssh = []
    labelssh = []
    vectorsli = []
    labelsli = []

    for dir in datadir:
        if dir.endswith('_{}_compl_aligned'.format(mode)):
            fulpa = data_folder + '/' + dir
            model = gensim.models.Word2Vec.load(fulpa)
            for word in words:
                name = word + str(re.search('\d{4}', dir).group())
                if word == 'friend':
                        vectorsfr.append(model.wv[word])
                        labelsfr.append(word+name)
                elif word == 'ship':
                        vectorssh.append(model.wv[word])
                        labelssh.append(word+name)
                elif word == 'like':
                        vectorsli.append(model.wv[word])
                        labelsli.append(word+name)
                else:
                    continue

    labelsfr = np.asarray(labelsfr)
    vectorsfr = np.asarray(vectorsfr)
    labelssh = np.asarray(labelssh)
    vectorssh = np.asarray(vectorssh)
    labelsli = np.asarray(labelsli)
    vectorsli = np.asarray(vectorsli)
    tsnef = TSNE(n_components=2, random_state=0)
    vectorsfr = tsnef.fit_transform(vectorsfr)
    tsnes = TSNE(n_components=2, random_state=1)
    vectorssh = tsnes.fit_transform(vectorssh)
    tsnel = TSNE(n_components=2, random_state=2)
    vectorsli = tsnel.fit_transform(vectorsli)

    x_vals_fr = [v[0] for v in vectorsfr]
    y_vals_fr = [v[1] for v in vectorsfr]
    x_vals_sh = [v[0] for v in vectorssh]
    y_vals_sh = [v[1] for v in vectorssh]
    x_vals_li = [v[0] for v in vectorsli]
    y_vals_li = [v[1] for v in vectorsli]

    if mode == 'o':
        namae = 'offline'
    elif mode == 'i':
        namae = 'Internet'

    matpl.figure(figsize=(10, 10))
    matpl.xlim(-size, size)
    matpl.ylim(-size, size)
    matpl.xlabel('Reduced Dimension 1', fontsize=18, fontweight='normal', labelpad=30)
    matpl.ylabel('Reduced Dimension 2', fontsize=18, fontweight='normal', va='bottom', ha='center')
    matpl.title('Distribution of Word Embeddings {}'.format(namae), fontsize=20, fontweight='normal', va='bottom')
    matpl.grid()
    coloursfr = ['yellow']
    labelsfr = list(labelsfr)
    colourssh = ['mediumturquoise']
    labelssh = list(labelssh)
    coloursli = ['mediumpurple']
    labelsli = list(labelsli)

    i = 0
    for label in sorted(labelsfr):
        noum = sorted(labelsfr).index(label)
        matpl.scatter(x_vals_fr[noum], y_vals_fr[noum], color=coloursfr[0], alpha=0.1*i+0.1)
        if noum == 9 and mode == 'o':
            matpl.annotate('friend', xy=(x_vals_fr[noum], y_vals_fr[noum]), fontsize=12, fontweight='light', ha='right', va='bottom')
        elif noum == 4 and mode == 'i':
            matpl.annotate('friend', xy=(x_vals_fr[noum], y_vals_fr[noum]), fontsize=12, fontweight='light', ha='right', va='bottom')
        else:
            matpl.annotate(str(noum+1), xy=(x_vals_fr[noum], y_vals_fr[noum]), fontsize=8, fontweight='light', ha='right', va='bottom')
            i += 9/10
    i = 0
    for label in sorted(labelssh):
        noum = sorted(labelssh).index(label)
        matpl.scatter(x_vals_sh[noum], y_vals_sh[noum], color=colourssh[0],  alpha=0.1*i+0.1)
        if noum == 9 and mode == 'o':
            matpl.annotate('ship', xy=(x_vals_sh[noum], y_vals_sh[noum]), fontsize=12, fontweight='light', ha='right',
                                   va='bottom')
        elif noum == 4 and mode == 'i':
            matpl.annotate('ship', xy=(x_vals_sh[noum], y_vals_sh[noum]), fontsize=12, fontweight='light', ha='right',
                                   va='bottom')
        else:
            matpl.annotate(str(noum+1), xy=(x_vals_sh[noum], y_vals_sh[noum]), fontsize=8, fontweight='light',
                                   ha='right',
                                   va='bottom')
            i += 9/10
    i = 0
    for label in sorted(labelsli):
        noum = sorted(labelsli).index(label)
        matpl.scatter(x_vals_li[noum], y_vals_li[noum], color=coloursli[0],  alpha=0.1*i+0.1)
        if noum == 9 and mode == 'o':
            matpl.annotate('like', xy=(x_vals_li[noum], y_vals_li[noum]), fontsize=12, fontweight='light', ha='right',
                                   va='bottom')
        elif noum == 4 and mode == 'i':
            matpl.annotate('like', xy=(x_vals_li[noum], y_vals_li[noum]), fontsize=12, fontweight='light', ha='right',
                                   va='bottom')
        else:
            matpl.annotate(str(noum + 1), xy=(x_vals_li[noum], y_vals_li[noum]), fontsize=8, fontweight='light',
                                   ha='right',
                                   va='bottom')
            i += 9/10

    figname = 'figure_Models_{}_{}_development'.format(namae, att)
    matpl.savefig(os.path.join(str(Path.cwd())+'/Graphs/'+figname))



def compare_cosine_sim(input, token: str, att: int, mode):
    """
    The models given as the input are loaded.
        Then, for all of the words on the wordlist, their similarity to the
        specified token is calculated for each point of data collection.
        The results are then plotted on a line graph to show the development of the similarities.
        """
    if os.path.isdir(input):
        modelllist = []
        comlist = os.listdir(str(Path.cwd())+'/'+input)
        for dir in comlist:
            if dir.endswith('_{}_compl_aligned'.format(mode)):
                modelllist.append(dir)

    timeline = ['1850', '1900', '1930s', '1961/62', '1991/92', '2004/05', '2006–8', '2012/13', '2017', '2020']

    modpa = str(Path.cwd())+'/Models aligned/'
    resdic = {}
    colours = ['lightcoral', 'turquoise', 'slateblue', 'palevioletred', 'palegreen', 'lightskyblue']

    modellist = sorted(modelllist)

    for word in words:
        if word != token:
            res = []
            for models in modellist:
                try:
                    compa = modpa + models
                    model = gensim.models.Word2Vec.load(compa)
                    res.append(model.wv.similarity(word, token))
                except KeyError:
                    res.append(0)
            resdic[word] = res

    figname = 'similarity_devel_{}_{}_{}'.format(token, att, mode)
    matpl.figure(figsize=(13, 11))
    matpl.ylim(-0.5, 0.8)
    matpl.xlim(0, 9.5)
    matpl.xlabel('Timeline', fontsize=24, fontweight='normal', labelpad=30)
    matpl.ylabel('Vector Similarity', fontsize=24, fontweight='normal', va='bottom', ha='center')
    matpl.xticks(range(0, 10), timeline, fontsize=16)
    matpl.plot(range(0, 11), [0,0,0,0,0,0,0,0,0,0,0], color='lightgray', linestyle='dotted')
    if mode == 'i':
        titel = token.capitalize() +' on the Internet'
        matpl.title(titel, fontsize=26, fontweight='normal', va='bottom')
    elif mode == 'o':
        titel = token.capitalize() + ' in Print/Newspapers'
        matpl.title(titel, fontsize=26, fontweight='normal', va='bottom')
    colpatch = []
    for key in resdic:
        i = list(resdic.keys()).index(key)
        value = resdic[key]
        if mode == 'i':
            matpl.plot(range(5, len(value)+5), value, colours[i])
        elif mode == 'o':
            matpl.plot(range(0, len(value)), value, colours[i])
        colpatch.append(mpat.Patch(color=colours[i], label=key))
    matpl.legend(handles=colpatch, fontsize=16)
    matpl.savefig(os.path.join(str(Path.cwd())+'/Graphs/'+figname))

def compare_cosine_sim_per_word(att: int):
    """
    For all of the words on the wordlist, the similarity between the two modes is calculated.
        The results are then plotted on a line graph to show the development of the similarities.
        """
    modpath = str(Path.cwd())+'/Models aligned/'
    offline = []
    internet = []

    for model in os.listdir(modpath):
        if model.endswith('o_compl_aligned'):
            offline.append(model)
        elif model.endswith('i_compl_aligned'):
            internet.append(model)

    offline = sorted(offline)[5:]
    internet = sorted(internet)
    resdic = {}
    colours = ['lightcoral', 'turquoise', 'slateblue', 'palevioletred', 'palegreen', 'lightskyblue']

    for word in words:
        res = []
        for num, model in enumerate(internet):
            try:
                print(model)
                m1 = modpath + model
                m2 = modpath + offline[num]
                print(m2)
                m1 = gensim.models.KeyedVectors.load(m1)
                v1 = m1.wv[word]
                v1 = v1.reshape(1, -1)
                m2 = gensim.models.KeyedVectors.load(m2)
                v2 = m2.wv[word]
                v2 = v2.reshape(1, -1)
                simi = pairwise.cosine_similarity(v1, v2)
                res.append(float(simi[0]))
            except KeyError:
                continue
        resdic[word] = res
    timeline = ['2004/5', '2006–8', '2012/13', '2017', '2020']
    figname = 'similarity_development_models_per_token_{}'.format(att)
    matpl.figure(figsize=(8, 8))
    matpl.ylim(-0.5, 0.8)
    matpl.xlim(0, 4)
    matpl.xticks(range(0, 5), timeline, fontsize=8)
    matpl.title('Model Similarity per Token and Year' ,fontsize=14, fontweight='normal', va='bottom')
    matpl.xlabel('Timeline', fontsize=18, fontweight='normal', labelpad=30)
    matpl.ylabel('Vector Similarity', fontsize=18, fontweight='normal', va='bottom', ha='center')
    matpl.plot(range(0, 11), [0,0,0,0,0,0,0,0,0,0,0], color='lightgray', linestyle='dotted')
    colpatch = []
    for key in resdic:
        if resdic[key] != []:
            print(resdic[key])
            i = list(resdic.keys()).index(key)
            value = resdic[key]
            matpl.plot(range(0, len(value)), value, colours[i])
            colpatch.append(mpat.Patch(color=colours[i], label=key))
    matpl.legend(handles=colpatch, fontsize=10)
    matpl.savefig(os.path.join(str(Path.cwd())+'/Graphs/'+figname))
