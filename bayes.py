#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import listdir
from os.path import isfile, join
import math
import random


def random_subset( iterator, K ):
    ''' From : http://stackoverflow.com/a/2612822 '''
    result = []
    N = 0

    for item in iterator:
        N += 1
        if len( result ) < K:
            result.append( item )
        else:
            s = int(random.random() * N)
            if s < K:
                result[ s ] = item

    return result

class Bayes:
    def __init__(self, type1, type2, training_ratio=0.8):
        self.type1 = type1
        self.type2 = type2
        self.training_ratio = training_ratio
        self.probs = {}
        self.nbarticles = {}
        self.nbwords = {}

    def train(self, folder, type_index):
        type = self.type1 if type_index == 1 else self.type2

        allfiles = [join(folder, f) for f in listdir(folder) if isfile(join(folder, f))]
        training_files = random_subset(allfiles, int(self.training_ratio * len(allfiles)))
        # FIXME: Texts are unicode, but some characters aren't, so open() crashes
        words_unfiltered = [line.split("\t") for line in [fd.readline() for fd in [open(file, 'r', encoding='ISO-8859-1') for file in training_files]]]
        words = self.filter_words(words_unfiltered)

        wordcount = {}
        self.nbarticles[type] = len(training_files)
        self.nbwords[type] = len(words)
        self.probs[type] = {}

        for word in words:
            try:
                wordcount[word] += 1
            except KeyError:
                wordcount[word] = 1

        for word in wordcount:
            self.probs[type][word] = (wordcount[word] + 1) /  float(len(words) + len(wordcount))
        #print(self.probs[type])

    def classify(self, filepath):
        words_unfiltered = [line.split("\t") for line in open(filepath, 'r', encoding='ISO-8859-1')]
        words = self.filter_words(words_unfiltered)

        wordcount = {}
        for word in words:
            try:
                wordcount[word] += 1
            except KeyError:
                wordcount[word] = 1

        p_1 = self.nbarticles[self.type1] / float(self.nbarticles[self.type1] + self.nbarticles[self.type2])
        p_2 = 1.0 - p_1

        for word in wordcount:
            try:
                l_prob_word_1 = self.probs[self.type1][word]
            except KeyError:
                l_prob_word_1 = 1.0 / float(self.nbwords[self.type1])
            try:
                l_prob_word_2 = self.probs[self.type2][word]
            except KeyError:
                l_prob_word_2 = 1.0 / float(self.nbwords[self.type2])

            p_1 *= math.log(pow(l_prob_word_1, wordcount[word]))
            p_2 *= math.log(pow(l_prob_word_2, wordcount[word]))

        return self.type1 if p_1 > p_2 else self.type2

    def filter_words(self, words):
        '''Takes a list of words in [original, type, canonical] form
        and returns a list of canonical words'''
        toret = []
        for word in words:
            try:
                if word[1] == 'NOM' or word[1] == 'ADJ' or word[1].startswith('VRB'):
                    toret.append(word[2].rstrip())
            except IndexError:
                pass

        return toret

    def test_algorithm(self, files_type1, files_type2):
        total_files = len(files_type1) + len(files_type2)
        correct_guesses_type1 = 0
        correct_guesses_type2 = 0
        for file_type1 in files_type1:
            if(self.classify(file_type1) == self.type1):
                correct_guesses_type1 += 1

        for file_type2 in files_type2:
            if(self.classify(file_type2) == self.type2):
                correct_guesses_type2 += 1

        return (correct_guesses_type1 + correct_guesses_type2) / float(total_files) * 100.0


if __name__ == '__main__':
    b = Bayes('pos', 'neg')
    b.train(sys.argv[1], 1)
    b.train(sys.argv[2], 2)

    folder_pos = sys.argv[2]
    allfiles = [join(folder_pos, f) for f in listdir(folder_pos) if isfile(join(folder_pos, f))]
    test_files_pos = random_subset(allfiles, int(0.2 * len(allfiles)))

    folder_neg = sys.argv[2]
    allfiles = [join(folder_neg, f) for f in listdir(folder_neg) if isfile(join(folder_neg, f))]
    test_files_neg = random_subset(allfiles, int(0.2 * len(allfiles)))

    result = b.test_algorithm(test_files_pos, test_files_neg)
    print("Percentage of successful classifications: %s%%" % result)