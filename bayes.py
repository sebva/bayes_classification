#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
from os import listdir
from os.path import isfile, join
import math
import random


class Bayes:
    def __init__(self, type1, type2):
        self.type1 = type1
        self.type2 = type2
        self.probs = {}
        self.nbarticles = {}
        self.nbwords = {}

    def train(self, training_files, type_index, is_tagged=True):
        current_type = self.type1 if type_index == 1 else self.type2

        files_lines = [fd.readlines() for fd in [open(file, 'r', encoding='UTF-8') for file in training_files]]
        words_unfiltered = []

        for file_lines in files_lines:
            for line in file_lines:
                if is_tagged:
                    words_unfiltered.append(line.split("\t"))
                else:
                    for word in line.split(' '):
                        if len(word) >= 2:
                            words_unfiltered.append(word.rstrip())

        if is_tagged:
            words = self.filter_words(words_unfiltered)
        else:
            words = words_unfiltered

        wordcount = {}
        self.nbarticles[current_type] = len(training_files)
        self.nbwords[current_type] = len(words)
        self.probs[current_type] = {}

        for word in words:
            try:
                wordcount[word] += 1
            except KeyError:
                wordcount[word] = 1

        for word in wordcount:
            self.probs[current_type][word] = (wordcount[word] + 1) / float(len(words) + len(wordcount))

    def classify(self, filepath, is_tagged=True):
        """

        :param filepath:string
        :return:string
        """
        if is_tagged:
            words_unfiltered = [line.split("\t") for line in open(filepath, 'r', encoding='UTF-8')]
        else:
            words_unfiltered = [line.rstrip().split(' ') for line in open(filepath, 'r', encoding='UTF-8')][0]

        if is_tagged:
            words = self.filter_words(words_unfiltered)
        else:
            words = words_unfiltered

        wordcount = {}
        for word in words:
            try:
                wordcount[word] += 1
            except KeyError:
                wordcount[word] = 1

        p_1 = self.nbarticles[self.type1] / float(self.nbarticles[self.type1] + self.nbarticles[self.type2])
        p_2 = math.log(1.0 - p_1)
        p_1 = math.log(p_1)

        # Counts different words from both sets, counting every same word once
        nb_different_words = len(set(self.probs[self.type1].keys()).union(set(self.probs[self.type2].keys())))
        empty_prob_type1 = 1.0 / float(self.nbwords[self.type1] + nb_different_words)
        empty_prob_type2 = 1.0 / float(self.nbwords[self.type2] + nb_different_words)

        for word in wordcount:
            try:
                l_prob_word_1 = self.probs[self.type1][word]
            except KeyError:
                l_prob_word_1 = empty_prob_type1
            try:
                l_prob_word_2 = self.probs[self.type2][word]
            except KeyError:
                l_prob_word_2 = empty_prob_type2

            p_1 += math.log(pow(l_prob_word_1, wordcount[word]))
            p_2 += math.log(pow(l_prob_word_2, wordcount[word]))

        return self.type1 if p_1 > p_2 else self.type2

    @staticmethod
    def filter_words(words):
        """
        Takes a list of words in [original, type, canonical] form
        and returns a list of canonical words
        :param words:list
        :return:list
        """
        toret = []
        for word in words:
            try:
                if word[1] == 'NOM' or word[1] == 'ADJ' or word[1].startswith('VER') or word[2] == '!':
                    toret.append(word[2].rstrip())
            except IndexError:
                pass

        return toret

    def test_algorithm(self, files_type1, files_type2, is_tagged=True):
        total_files = len(files_type1) + len(files_type2)
        correct_guesses_type1 = 0
        correct_guesses_type2 = 0
        for file_type1 in files_type1:
            if self.classify(file_type1, is_tagged) == self.type1:
                correct_guesses_type1 += 1

        for file_type2 in files_type2:
            if self.classify(file_type2, is_tagged) == self.type2:
                correct_guesses_type2 += 1

        return (correct_guesses_type1 + correct_guesses_type2) / float(total_files) * 100.0


if __name__ == '__main__':
    b = Bayes('pos', 'neg')

    training_ratio = 0.8

    folder_pos = sys.argv[1]
    folder_neg = sys.argv[2]

    is_tagged = True if len(sys.argv) > 3 and sys.argv[3] == 'tagged' else False

    pos_files = [join(folder_pos, f) for f in listdir(folder_pos) if isfile(join(folder_pos, f))]
    neg_files = [join(folder_neg, f) for f in listdir(folder_neg) if isfile(join(folder_neg, f))]

    random.shuffle(pos_files)
    random.shuffle(neg_files)

    b.train(pos_files[:int(len(pos_files) * training_ratio)], 1, is_tagged)
    b.train(neg_files[:int(len(neg_files) * training_ratio)], 2, is_tagged)

    test_files_pos = pos_files[int(len(pos_files) * training_ratio):]
    test_files_neg = neg_files[int(len(neg_files) * training_ratio):]

    result = b.test_algorithm(test_files_pos, test_files_neg, is_tagged)
    print("Percentage of successful classifications: %f%%" % result)