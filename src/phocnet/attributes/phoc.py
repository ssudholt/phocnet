'''
Created on Dec 17, 2015

@author: ssudholt
'''
import logging

import numpy as np

def get_most_common_n_grams(words, num_results=50, n=2):
    '''
    Calculates the 50 (default) most common bigrams (default) from a
    list of pages, where each page is a list of WordData objects.

    Args:
        words (list of str): List containing the word strings from which to extract the bigrams
        num_results (int): Number of n-grams returned.
        n (int): length of n-grams.
    Returns:
        most common <n>-grams
    '''
    ngrams = {}
    for w in words:
        w_ngrams = get_n_grams(w, n)
        for ng in w_ngrams:
            ngrams[ng] = ngrams.get(ng, 0) + 1
    sorted_list = sorted(ngrams.items(), key=lambda x: x[1], reverse=True)
    top_ngrams = sorted_list[:num_results]
    return {k: i for i, (k, _) in enumerate(top_ngrams)}

def get_n_grams(word, n):
    '''
    Calculates list of ngrams for a given word.

    Args:
        word (str): Word to calculate ngrams for.
        n (int): Maximal ngram size: n=3 extracts 1-, 2- and 3-grams.
    Returns:
        List of ngrams as strings.
    '''
    return [word[i:i+n]for i in range(len(word)-n+1)]
    
def build_phoc(words, phoc_unigrams, unigram_levels, 
               bigram_levels=None, phoc_bigrams=None,
               split_character=None, on_unknown_unigram='error'):
    '''
    Calculate Pyramidal Histogram of Characters (PHOC) descriptor (see Almazan 2014).

    Args:
        word (str): word to calculate descriptor for
        phoc_unigrams (str): string of all unigrams to use in the PHOC
        unigram_levels (list of int): the levels for the unigrams in PHOC
        phoc_bigrams (list of str): list of bigrams to be used in the PHOC
        phoc_bigram_levls (list of int): the levels of the bigrams in the PHOC
        split_character (str): special character to split the word strings into characters
        on_unknown_unigram (str): What to do if a unigram appearing in a word
            is not among the supplied phoc_unigrams. Possible: 'warn', 'error'
    Returns:
        the PHOC for the given word
    '''
    # prepare output matrix
    logger = logging.getLogger('PHOCGenerator')
    if on_unknown_unigram not in ['error', 'warn']:
        raise ValueError('I don\'t know the on_unknown_unigram parameter \'%s\'' % on_unknown_unigram)
    phoc_size = len(phoc_unigrams) * np.sum(unigram_levels)
    if phoc_bigrams is not None:
        phoc_size += len(phoc_bigrams)*np.sum(bigram_levels)
    phocs = np.zeros((len(words), phoc_size))
    # prepare some lambda functions
    occupancy = lambda k, n: [float(k) / n, float(k+1) / n]
    overlap = lambda a, b: [max(a[0], b[0]), min(a[1], b[1])]
    size = lambda region: region[1] - region[0]
    
    # map from character to alphabet position
    char_indices = {d: i for i, d in enumerate(phoc_unigrams)}
    
    # iterate through all the words
    for word_index, word in enumerate(words):        
        if split_character is not None:
            word = word.split(split_character)       

	n = len(word) 
        for index, char in enumerate(word):
            char_occ = occupancy(index, n)
            if char not in char_indices:
                if on_unknown_unigram == 'warn':
                    logger.warn('The unigram \'%s\' is unknown, skipping this character', char)
                    continue
                else:
                    logger.fatal('The unigram \'%s\' is unknown', char)
                    raise ValueError()
            char_index = char_indices[char]
            for level in unigram_levels:
                for region in range(level):
                    region_occ = occupancy(region, level)
                    if size(overlap(char_occ, region_occ)) / size(char_occ) >= 0.5:
                        feat_vec_index = sum([l for l in unigram_levels if l < level]) * len(phoc_unigrams) + region * len(phoc_unigrams) + char_index
                        phocs[word_index, feat_vec_index] = 1
        # add bigrams
        if phoc_bigrams is not None:
            ngram_features = np.zeros(len(phoc_bigrams)*np.sum(bigram_levels))
            ngram_occupancy = lambda k, n: [float(k) / n, float(k+2) / n]
            for i in range(n-1):
                ngram = word[i:i+2]
                if phoc_bigrams.get(ngram, 0) == 0:
                    continue
                occ = ngram_occupancy(i, n)
                for level in bigram_levels:
                    for region in range(level):
                        region_occ = occupancy(region, level)
                        overlap_size = size(overlap(occ, region_occ)) / size(occ)
                        if overlap_size >= 0.5:
                            ngram_features[region * len(phoc_bigrams) + phoc_bigrams[ngram]] = 1
            phocs[word_index, -ngram_features.shape[0]:] = ngram_features        
    return phocs

def unigrams_from_word_list(word_list, split_character=None):
    if split_character is not None:
        unigrams = [elem for word in word_list for elem in word.get_transcription().split(split_character)]
    else:
        unigrams = [elem for word in word_list for elem in word.get_transcription()]
    unigrams = list(sorted(set(unigrams)))
    return unigrams
