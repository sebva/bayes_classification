#!/usr/bin/env python
# -*- coding: utf-8 -*-
from os import listdir
from os.path import isfile, join
import math
import random


class Bayes:
    """
    Bayesian classifier by Sébastien Vaucher and Jason Racine
    """
    def __init__(self, type1, type2, is_tagged):
        """
        Initialize Bayesian classifier
        :param type1:str The name of the first category
        :param type2:str The name of the second category
        :param is_tagged:bool Whether the articles that will be processed are tagged or not
        """
        self.type1 = type1
        self.type2 = type2
        self.is_tagged = is_tagged
        self.probs = {}
        self.nbarticles = {}
        self.nbwords = {}
        self.nb_different_words = -1

    def train(self, training_files, type_index):
        """
        Train the Bayesian classifier
        :param training_files:list A list of file paths from articles of the same category
        :param type_index:str The name of the category, must be one of the categories passed in the initializer
        """
        current_type = self.type1 if type_index == 1 else self.type2

        # Create a list containing a list of lines
        files_lines = [fd.readlines() for fd in [open(file, 'r', encoding='UTF-8') for file in training_files]]
        words_unfiltered = []

        # Consolidation of a global word list
        for file_lines in files_lines:
            for line in file_lines:
                if self.is_tagged:
                    words_unfiltered.append(line.split("\t"))
                else:
                    for word in line.split(' '):
                        if len(word) >= 2:
                            words_unfiltered.append(word.rstrip())

        words = self.filter_words(words_unfiltered)

        wordcount = {}
        self.nbarticles[current_type] = len(training_files)
        self.nbwords[current_type] = len(words)
        self.probs[current_type] = {}

        # Count occurrences of every different word
        for word in words:
            try:
                wordcount[word] += 1
            except KeyError:
                wordcount[word] = 1

        # Compute the bayesian probability
        for word in wordcount:
            self.probs[current_type][word] = (wordcount[word] + 1) / float(len(words) + len(wordcount))

    def classify(self, filepath):
        """
        Classify an article
        :param filepath:str The path where the article lies on the file system
        :return: The name of the category to which the classifier thinks the article belongs to
        :rtype : str
        """
        # Create a list of every word in the document
        if self.is_tagged:
            words_unfiltered = [line.split("\t") for line in open(filepath, 'r', encoding='UTF-8')]
        else:
            words_unfiltered = [line.rstrip().split(' ') for line in open(filepath, 'r', encoding='UTF-8')][0]

        words = self.filter_words(words_unfiltered)

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
        nb_different_words = self.size_bayesian_network()
        # Compute the zero probability
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

    def size_bayesian_network(self):
        """
        Get the number of different words found during training.
        Training must be called prior to calling this method
        :return: The number of different words known by the classifier
        :rtype : int
        """
        # We keep the value found to make the algorithm run faster
        if self.nb_different_words == -1:
            self.nb_different_words = len(set(self.probs[self.type1].keys()).union(set(self.probs[self.type2].keys())))

        return self.nb_different_words

    def filter_words(self, words):
        """
        Takes a list of words and returns a filtered list of canonical words.
        Words are in [original, type, canonical] form if tagged, or just str if untagged
        :param words:list The list of words to filter
        :return: A filtered list of canonical words
        :rtype : list
        """
        toret = []
        for word in words:
            try:
                if self.is_tagged:
                    if word[1] == 'NOM' or word[1] == 'ADJ' or word[1].startswith('VER') or word[2] == '!':
                        toret.append(word[2].rstrip())
                else:
                    if len(word) > 3:
                        toret.append(word.rstrip())
            except IndexError:
                pass

        return toret

    def test_algorithm(self, files_type1, files_type2):
        """
        Test the algorithm by classifying a list of articles
        :param files_type1:list A list of file paths of articles known as being of category number 1
        :param files_type2:list A list of file paths of articles known as being of category number 2
        :return: A tuple of 2 values, each being the percentage of successful classification of a category
        :rtype : tuple
        """
        correct_guesses_type1 = 0
        correct_guesses_type2 = 0
        for file_type1 in files_type1:
            if self.classify(file_type1) == self.type1:
                correct_guesses_type1 += 1

        for file_type2 in files_type2:
            if self.classify(file_type2) == self.type2:
                correct_guesses_type2 += 1

        return correct_guesses_type1 / len(files_type1) * 100.0, correct_guesses_type2 / len(files_type2) * 100.0

    @staticmethod
    def cross_validate(files_type1, files_type2, k_divisions, is_tagged):
        """
        Perform K-Fold Cross-Validation on the algorithm
        :param files_type1:list A list of files known as being of category 1
        :param files_type2:list A list of files known as being of category 2
        :param k_divisions:int The number of divisions, known as k in k-fold cross-validation
        :param is_tagged:bool Whether the articles are tagged or not
        :return: A list of 2 values, each being the mean percentage of successful classification of a category
        :rtype : list
        """
        total_results = [0, 0]

        # Divide the file lists in k equal chunks
        chunk_size1 = int(len(files_type1) / k_divisions)
        corpuses_type1 = [files_type1[i:i + chunk_size1] for i in range(0, len(files_type1), chunk_size1)]
        chunk_size2 = int(len(files_type2) / k_divisions)
        corpuses_type2 = [files_type2[i:i + chunk_size2] for i in range(0, len(files_type2), chunk_size2)]

        # i = index of the test corpus
        for i in range(k_divisions):
            b = Bayes('pos', 'neg', is_tagged)

            train_files_type1 = []
            train_files_type2 = []
            for j in range(k_divisions):
                if j != i:
                    train_files_type1.extend(corpuses_type1[j])
                    train_files_type2.extend(corpuses_type2[j])

            b.train(train_files_type1, 1)
            b.train(train_files_type2, 2)

            res = b.test_algorithm(corpuses_type1[i], corpuses_type2[i])
            total_results[0] += res[0]
            total_results[1] += res[1]

        return [total_result / k_divisions for total_result in total_results]

if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Bayesian classifier by Sébastien Vaucher and Jason Racine')
    parser.add_argument('folder_pos', default=None, help='Folder containing positive articles')
    parser.add_argument('folder_neg', default=None, help='Folder containing negative articles')
    parser.add_argument('-t', '--tagged', default=False, action='store_true',
                        help='Tell the program that the articles are already tagged')
    parser.add_argument('-c', '--cross', default=False, action='store_true',
                        help='Perform K-fold cross-validation on the algorithm')
    parser.add_argument('-k', '--k-divisions', type=int, default=5,
                        help='Number of divisions for K-fold cross-validation')
    parser.add_argument('-r', '--training-ratio', type=float, default=0.8,
                        help='Ratio of articles used in the training phase')

    args = parser.parse_args()

    # Create file lists
    pos_files = [join(args.folder_pos, f) for f in listdir(args.folder_pos) if isfile(join(args.folder_pos, f))]
    neg_files = [join(args.folder_neg, f) for f in listdir(args.folder_neg) if isfile(join(args.folder_neg, f))]

    random.shuffle(pos_files)
    random.shuffle(neg_files)

    if args.cross:
        result = Bayes.cross_validate(pos_files, neg_files, args.k_divisions, args.tagged)
    else:
        b = Bayes('pos', 'neg', args.tagged)
        b.train(pos_files[:int(len(pos_files) * args.training_ratio)], 1)
        b.train(neg_files[:int(len(neg_files) * args.training_ratio)], 2)

        print("Bayesian network size: %d" % b.size_bayesian_network())

        test_files_pos = pos_files[int(len(pos_files) * args.training_ratio):]
        test_files_neg = neg_files[int(len(neg_files) * args.training_ratio):]

        result = b.test_algorithm(test_files_pos, test_files_neg)

    print("Percentage of successful classifications (pos): %.1f%%" % result[0])
    print("Percentage of successful classifications (neg): %.1f%%" % result[1])