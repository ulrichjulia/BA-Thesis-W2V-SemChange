"""Many thanks for the publicly available initial code script written by Ryan Heuser (https://gist.github.com/quadrismegistus/09a93e219a6ffc4f216fb85235535faf).
Adapted for Python 3.8 and gensim 3.8.3 by Julia Ulrich."""

from pathlib import Path
import gensim.models
import numpy as np
import os

def smart_procrustes_align_gensim(base_embed, other_embed, name, words=None):
	"""Procrustes align two gensim word2vec models (to allow for comparison between same word across models).
	Code ported from HistWords <https://github.com/williamleif/histwords> by William Hamilton <wleif@stanford.edu>.
		(With help from William. Thank you!)

	First, intersect the vocabularies (see `intersection_align_gensim` documentation).
	Then do the alignment on the other_embed model.
	Replace the other_embed model's syn0 and syn0norm numpy matrices with the aligned version.
	Return other_embed.

	If `words` is set, intersect the two models' vocabulary with the vocabulary in words (see `intersection_align_gensim` documentation).
	"""

	modpa = str(Path.cwd())+'/Models unaligned/'
	base_embed, other_embed = gensim.models.Word2Vec.load(modpa+base_embed), gensim.models.Word2Vec.load(modpa+other_embed)

	base_embed.wv.init_sims()
	other_embed.wv.init_sims()

	# get the embedding matrices
	base_vecs = base_embed.wv.vectors_norm
	other_vecs = other_embed.wv.vectors_norm

	# just a matrix dot product with numpy
	m = other_vecs.T.dot(base_vecs) 
	# SVD method from numpy
	u, _, v = np.linalg.svd(m)
	# another matrix operation
	ortho = u.dot(v) 
	# Replace original array with modified one
	# i.e. multiplying the embedding matrix (syn0norm)by "ortho"
	other_embed.wv.vectors_norm = other_embed.wv.vectors = (other_embed.wv.vectors_norm).dot(ortho)
	other_embed.save(os.path.join(str(Path.cwd())+'/Models aligned/'+'{}_compl_aligned'.format(name)))
	return other_embed
	
def intersection_align_gensim(m1,m2, words=None):
	"""
	Intersect two gensim word2vec models, m1 and m2.
	Only the shared vocabulary between them is kept.
	If 'words' is set (as list or set), then the vocabulary is intersected with this list as well.
	Indices are re-organized from 0..N in order of descending frequency (=sum of counts from both m1 and m2).
	These indices correspond to the new syn0 and syn0norm objects in both gensim models:
		-- so that Row 0 of m1.syn0 will be for the same word as Row 0 of m2.syn0
		-- you can find the index of any word on the .index2word list: model.index2word.index(word) => 2
	The .vocab dictionary is also updated for each model, preserving the count but updating the index.
	"""
	nam1, nam2 = str(m1), str(m2)
	namlist = [nam1, nam2]

	path1 = str(Path.cwd())+'/Models unaligned/'+m1
	path2 = str(Path.cwd())+'/Models unaligned/'+m2

	m1 = gensim.models.Word2Vec.load(path1)
	m2 = gensim.models.Word2Vec.load(path2)

	# Get the vocab for each model
	vocab_m1 = set(m1.wv.vocab)
	vocab_m2 = set(m2.wv.vocab)

	# Find the common vocabulary
	common_vocab = []
	for word in vocab_m1:
		if word in vocab_m2:
			common_vocab.append(word)
	common_vocab = set(common_vocab)


	# If no alignment necessary because vocab is identical...
	if len(vocab_m1)/len(common_vocab) == 1:
		print('No alignemt done because the models are the same.')
		return (m1,m2)

	# Otherwise sort by frequency (summed for both)
	common_vocab = list(common_vocab)
	common_vocab.sort(key=lambda w: m1.wv.vocab[w].count + m2.wv.vocab[w].count,reverse=True)

	modli = [m1,m2]
	# Then for each model...
	for m in modli:
		# Replace old syn0norm array with new one (with common vocab)
		indices = [m.wv.vocab[w].index for w in common_vocab]
		old_arr = m.wv.vectors
		new_arr = np.array([old_arr[index] for index in indices])
		m.wv.vectors_norm = m.wv.vectors = new_arr

		# Replace old vocab dictionary with new one (with common vocab)
		# and old index2word with new one
		m.wv.index2word = common_vocab
		old_vocab = m.wv.vocab
		new_vocab = {}
		for new_index, word in enumerate(common_vocab):
			old_vocab_obj = old_vocab[word]
			new_vocab[word] = gensim.models.word2vec.Vocab(index=new_index, count=old_vocab_obj.count)
		m.wv.vocab = new_vocab
		m.save(os.path.join(str(Path.cwd())+'/Models unaligned/'+'{}_vocab_corrected'.format(namlist[modli.index(m)])))
	print('Intersection alignment worked.')
	return (m1,m2)

