import pickle
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
from fuzzywuzzy import fuzz
from intervaltree import IntervalTree
from scipy.stats import spearmanr
from sklearn.base import BaseEstimator
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import make_scorer, matthews_corrcoef
from sklearn.model_selection import StratifiedKFold, cross_validate
from sklearn.pipeline import make_pipeline
from sklearn.utils.validation import check_array, check_is_fitted, check_X_y

import stop
from lasearch.db import FlatFileDatabase


def _main():
    db = FlatFileDatabase(db_dir=Path('flatdb'))
    latin_tess_files_dir = Path('data')
    aeneid_path = latin_tess_files_dir / 'vergil.aeneid.tess'
    lucan_file = 'lucan.bellum_civile.part.1.tess'
    lucan1_path = latin_tess_files_dir / lucan_file
    aeneid_data = db.retrieve(aeneid_path)
    lucan_data = db.retrieve(lucan1_path)
    parallels = _read_benchmark()

    results = []
    for name, scorer in {
            'Rescaled Max Word Factor Norm':
            scorer_builder(_score_by_rescaled_max_word_factor_norm,
                           'scoreds_c.pickle'),
            'Rescaled Max Diff':
            scorer_builder(_score_by_rescaled_max_diff, 'scoreds_d.pickle'),
            'Threshold and Rescaling':
            scorer_builder(_score_by_threshold_and_rescaling,
                           'scoreds_e.pickle'),
            'Threshold and Rescaled Max Word':
            scorer_builder(_score_by_threshold_and_rescaled_max_word,
                           'scoreds_f.pickle'),
            'With Stopwords':
            scorer_builder(_score_with_stopwords, 'scoreds_g.pickle'),
            'With Stopwords and Threshold':
            scorer_builder(_score_with_stopwords_and_threshold,
                           'scoreds_h.pickle'),
            'With Stopwords and Lemmata':
            scorer_builder(_score_with_stopwords_and_lemmata,
                           'scoreds_i.pickle'),
            'Mean Pool Cos Sim':
            scorer_builder(_score_by_mean_pool_cos_sim, 'scoreds_j.pickle')
    }.items():
        similarities = _compute_similarities(aeneid_data, lucan_data)
        scoreds = scorer(aeneid_data, lucan_data, similarities)
        print('-' * 20)
        print(name)
        print('-' * 20)
        # _print_scoreds(aeneid_data, lucan_data, scoreds)
        results.append(_evaluate(aeneid_data, lucan_data, scoreds, parallels))
    # _plot(results)


def scorer_builder(f, outpathstr):

    def f_caller(aeneid_data, lucan_data, similarities):
        outpath = Path(outpathstr)
        if outpath.exists():
            with outpath.open('rb') as ifh:
                return pickle.load(ifh)
        start = time.time()
        scoreds = f(aeneid_data, lucan_data, similarities)
        score_time = time.time() - start
        print('Score time:', score_time)
        print('Total scored:', len(scoreds))
        with outpath.open('wb') as ofh:
            pickle.dump(scoreds, ofh)
        return scoreds

    return f_caller


def _compute_similarities(aeneid_data, lucan_data):
    outpath = Path('mult.npy')
    if outpath.exists():
        return np.load(str(outpath))
    start = time.time()
    normed_aeneid = aeneid_data.embeddings / np.linalg.norm(
        aeneid_data.embeddings, axis=1)[:, np.newaxis]
    normed_lucan = lucan_data.embeddings / np.linalg.norm(
        lucan_data.embeddings, axis=1)[:, np.newaxis]
    similarities = np.matmul(normed_aeneid, normed_lucan.T)
    compute_time = time.time() - start
    print('Compute time:', compute_time)
    print('Aeneid shape:', normed_aeneid.shape)
    print('Lucan shape:', normed_lucan.shape)

    with open(str(outpath), 'wb') as ofh:
        np.save(ofh, similarities)

    return similarities


def _score(aeneid_data, lucan_data, similarities):
    outpath = Path('scoreds.pickle')
    if outpath.exists():
        with open(str(outpath), 'rb') as ifh:
            return pickle.load(ifh)
    start = time.time()
    # scoreds = _score_by_sum(aeneid_data, lucan_data, similarities)
    # scoreds = _score_by_cls(aeneid_data, lucan_data, similarities)
    # scoreds = _score_by_word_sum(aeneid_data, lucan_data, similarities)
    # scoreds = _score_by_scaling(aeneid_data, lucan_data, similarities)
    # scoreds = _score_by_threshold(aeneid_data, lucan_data, similarities)
    # scoreds = _score_by_threshold_and_scaling(aeneid_data, lucan_data,
    # similarities)
    # scoreds = _score_by_rescaling(aeneid_data, lucan_data, similarities)
    # scoreds = _score_by_threshold_and_rescaling(aeneid_data, lucan_data,
    # similarities)
    # scoreds = _score_by_rescaled_word_sum(aeneid_data, lucan_data,
    # similarities)
    scoreds = _score_by_rescaled_max_word(aeneid_data, lucan_data,
                                          similarities)

    score_time = time.time() - start
    print('Score time:', score_time)
    print('Total scored:', len(scoreds))

    with open('scoreds.pickle', 'wb') as ofh:
        pickle.dump(scoreds, ofh)

    return scoreds


def _score_by_sum(aeneid_data, lucan_data, similarities):
    scoreds = []
    aeneid_bounds = _get_sentence_boundaries(aeneid_data)
    lucan_bounds = _get_sentence_boundaries(lucan_data)
    for i, (a_start,
            a_end) in enumerate(zip(aeneid_bounds[:-1], aeneid_bounds[1:])):
        for j, (l_start,
                l_end) in enumerate(zip(lucan_bounds[:-1], lucan_bounds[1:])):
            scoreds.append((similarities[a_start:a_end, l_start:l_end].sum() /
                            ((a_end - a_start) + (l_end - l_start)), i, j))
    scoreds.sort(reverse=True)
    return scoreds


def _score_by_cls(aeneid_data, lucan_data, similarities):
    scoreds = []
    aeneid_bounds = _get_sentence_boundaries(aeneid_data)
    lucan_bounds = _get_sentence_boundaries(lucan_data)
    for i, a_start in enumerate(aeneid_bounds[:-1]):
        for j, l_start in enumerate(lucan_bounds[:-1]):
            scoreds.append((similarities[a_start, l_start], i, j))
    scoreds.sort(reverse=True)
    return scoreds


def _score_by_word_sum(aeneid_data, lucan_data, similarities):
    """NB: mutates ``similarities``

    This seems to choose the longest sentences. Adjusting normalization of
    scores to be the multiplication of the sentence lengths goes too far and
    chooses the shortest sentences.
    """
    scoreds = []
    # zero out non-word similarities
    aeneid_non_word_indices = _get_non_word_indices(aeneid_data)
    lucan_non_word_indices = _get_non_word_indices(lucan_data)
    similarities[aeneid_non_word_indices, :] = 0
    similarities[:, lucan_non_word_indices] = 0
    aeneid_bounds = _get_sentence_boundaries(aeneid_data)
    lucan_bounds = _get_sentence_boundaries(lucan_data)
    for i, (a_start,
            a_end) in enumerate(zip(aeneid_bounds[:-1], aeneid_bounds[1:])):
        for j, (l_start,
                l_end) in enumerate(zip(lucan_bounds[:-1], lucan_bounds[1:])):
            # ignore initial [CLS] and final [SEP]
            scoreds.append(
                (similarities[a_start + 1:a_end - 1,
                              l_start + 1:l_end - 1].sum() /
                 ((a_end - a_start - 2) + (l_end - l_start - 2)), i, j))
    scoreds.sort(reverse=True)
    return scoreds


def _score_by_scaling(aeneid_data, lucan_data, similarities):
    """Adjust simliarities by inverse frequencies of tokens compared"""
    scoreds = []
    sqrt_aeneid_anti_freqs = np.sqrt(1 - aeneid_data.get_token_frequencies())
    sqrt_lucan_anti_freqs = np.sqrt(1 - lucan_data.get_token_frequencies())
    aeneid_bounds = _get_sentence_boundaries(aeneid_data)
    lucan_bounds = _get_sentence_boundaries(lucan_data)
    for i, (a_start,
            a_end) in enumerate(zip(aeneid_bounds[:-1], aeneid_bounds[1:])):
        for j, (l_start,
                l_end) in enumerate(zip(lucan_bounds[:-1], lucan_bounds[1:])):
            # taking the sqrt of an outer product is equal to taking the outer
            # product of the sqrts
            factors = np.outer(sqrt_aeneid_anti_freqs[a_start:a_end],
                               sqrt_lucan_anti_freqs[l_start:l_end])
            score = np.multiply(similarities[a_start:a_end, l_start:l_end],
                                factors).sum()
            comparisons_count = (a_end - a_start) * (l_end - l_start)
            scoreds.append((score / comparisons_count, i, j))
    scoreds.sort(reverse=True)
    return scoreds


def _score_by_threshold(aeneid_data, lucan_data, similarities):
    """NB: mutates similarities; Count only if similarity is in top 75%"""
    scoreds = []
    threshold = np.quantile(similarities.ravel(), 0.75)
    mask = similarities >= threshold
    similarities[~mask] = 0
    aeneid_bounds = _get_sentence_boundaries(aeneid_data)
    lucan_bounds = _get_sentence_boundaries(lucan_data)
    for i, (a_start,
            a_end) in enumerate(zip(aeneid_bounds[:-1], aeneid_bounds[1:])):
        for j, (l_start,
                l_end) in enumerate(zip(lucan_bounds[:-1], lucan_bounds[1:])):
            comparisons_count = mask[a_start:a_end, l_start:l_end].sum()
            if comparisons_count == 0:
                continue
            score = similarities[a_start:a_end, l_start:l_end].sum()
            scoreds.append((score / comparisons_count, i, j))
    scoreds.sort(reverse=True)
    return scoreds


def _score_by_threshold_and_scaling(aeneid_data, lucan_data, similarities):
    """NB: mutates similarities;

    Adjust top x% simliarities by inverse frequencies of tokens compared
    """
    scoreds = []
    threshold = np.quantile(similarities.ravel(), 0.75)
    mask = similarities >= threshold
    similarities[~mask] = 0
    sqrt_aeneid_anti_freqs = np.sqrt(1 - aeneid_data.get_token_frequencies())
    sqrt_lucan_anti_freqs = np.sqrt(1 - lucan_data.get_token_frequencies())
    aeneid_bounds = _get_sentence_boundaries(aeneid_data)
    lucan_bounds = _get_sentence_boundaries(lucan_data)
    for i, (a_start,
            a_end) in enumerate(zip(aeneid_bounds[:-1], aeneid_bounds[1:])):
        for j, (l_start,
                l_end) in enumerate(zip(lucan_bounds[:-1], lucan_bounds[1:])):
            comparisons_count = mask[a_start:a_end, l_start:l_end].sum()
            if comparisons_count == 0:
                continue
            # taking the sqrt of an outer product is equal to taking the outer
            # product of the sqrts
            factors = np.outer(sqrt_aeneid_anti_freqs[a_start:a_end],
                               sqrt_lucan_anti_freqs[l_start:l_end])
            score = np.multiply(similarities[a_start:a_end, l_start:l_end],
                                factors).sum()
            scoreds.append((score / comparisons_count, i, j))
    scoreds.sort(reverse=True)
    return scoreds


def _score_by_rescaling(aeneid_data, lucan_data, similarities):
    """Adjust simliarities by scale frequencies of tokens compared"""
    scoreds = []
    aeneid_freqs = aeneid_data.get_token_frequencies()
    lucan_freqs = lucan_data.get_token_frequencies()
    sqrt_aeneid_anti_freqs = np.sqrt(1 - (aeneid_freqs / aeneid_freqs.max()))
    sqrt_lucan_anti_freqs = np.sqrt(1 - (lucan_freqs / lucan_freqs.max()))
    aeneid_bounds = _get_sentence_boundaries(aeneid_data)
    lucan_bounds = _get_sentence_boundaries(lucan_data)
    for i, (a_start,
            a_end) in enumerate(zip(aeneid_bounds[:-1], aeneid_bounds[1:])):
        for j, (l_start,
                l_end) in enumerate(zip(lucan_bounds[:-1], lucan_bounds[1:])):
            # taking the sqrt of an outer product is equal to taking the outer
            # product of the sqrts
            factors = np.outer(sqrt_aeneid_anti_freqs[a_start:a_end],
                               sqrt_lucan_anti_freqs[l_start:l_end])
            score = np.multiply(similarities[a_start:a_end, l_start:l_end],
                                factors).sum()
            comparisons_count = (a_end - a_start) * (l_end - l_start)
            scoreds.append((score / comparisons_count, i, j))
    scoreds.sort(reverse=True)
    return scoreds


def _score_by_threshold_and_rescaling(aeneid_data, lucan_data, similarities):
    """NB: mutates similarities;

    Adjust top x% simliarities by rescaled frequencies of tokens compared
    """
    scoreds = []
    threshold = np.quantile(similarities.ravel(), 0.9)
    mask = similarities >= threshold
    similarities[~mask] = 0
    aeneid_freqs = aeneid_data.get_token_frequencies()
    lucan_freqs = lucan_data.get_token_frequencies()
    sqrt_aeneid_anti_freqs = np.sqrt(1 - (aeneid_freqs / aeneid_freqs.max()))
    sqrt_lucan_anti_freqs = np.sqrt(1 - (lucan_freqs / lucan_freqs.max()))
    aeneid_bounds = _get_sentence_boundaries(aeneid_data)
    lucan_bounds = _get_sentence_boundaries(lucan_data)
    for i, (a_start,
            a_end) in enumerate(zip(aeneid_bounds[:-1], aeneid_bounds[1:])):
        for j, (l_start,
                l_end) in enumerate(zip(lucan_bounds[:-1], lucan_bounds[1:])):
            comparisons_count = mask[a_start:a_end, l_start:l_end].sum()
            if comparisons_count == 0:
                continue
            # taking the sqrt of an outer product is equal to taking the outer
            # product of the sqrts
            factors = np.outer(sqrt_aeneid_anti_freqs[a_start:a_end],
                               sqrt_lucan_anti_freqs[l_start:l_end])
            score = np.multiply(similarities[a_start:a_end, l_start:l_end],
                                factors).sum()
            scoreds.append((score / comparisons_count, i, j))
    scoreds.sort(reverse=True)
    return scoreds


def _score_by_rescaled_word_sum(aeneid_data, lucan_data, similarities):
    """Adjust simliarities by scaled frequencies of word tokens compared

    NB: mutates ``similarities``
    """
    scoreds = []
    aeneid_freqs = aeneid_data.get_token_frequencies()
    lucan_freqs = lucan_data.get_token_frequencies()
    sqrt_aeneid_anti_freqs = np.sqrt(1 - (aeneid_freqs / aeneid_freqs.max()))
    sqrt_lucan_anti_freqs = np.sqrt(1 - (lucan_freqs / lucan_freqs.max()))
    aeneid_non_word_indices = _get_non_word_indices(aeneid_data)
    lucan_non_word_indices = _get_non_word_indices(lucan_data)
    similarities[aeneid_non_word_indices, :] = 0
    similarities[:, lucan_non_word_indices] = 0
    aeneid_bounds = _get_sentence_boundaries(aeneid_data)
    lucan_bounds = _get_sentence_boundaries(lucan_data)
    for i, (a_start_raw,
            a_end_raw) in enumerate(zip(aeneid_bounds[:-1],
                                        aeneid_bounds[1:])):
        # ignore initial [CLS] and final [SEP]
        a_start = a_start_raw + 1
        a_end = a_end_raw - 1
        for j, (l_start_raw, l_end_raw) in enumerate(
                zip(lucan_bounds[:-1], lucan_bounds[1:])):
            l_start = l_start_raw + 1
            l_end = l_end_raw - 1
            # taking the sqrt of an outer product is equal to taking the outer
            # product of the sqrts
            factors = np.outer(sqrt_aeneid_anti_freqs[a_start:a_end],
                               sqrt_lucan_anti_freqs[l_start:l_end])
            score = np.multiply(similarities[a_start:a_end, l_start:l_end],
                                factors).sum()
            comparisons_count = (a_end - a_start) * (l_end - l_start)
            scoreds.append((score / comparisons_count, i, j))
    scoreds.sort(reverse=True)
    return scoreds


def _score_by_rescaled_max_word(aeneid_data, lucan_data, similarities):
    """Adjust simliarities by scaled frequencies of highest simliarity word
    tokens compared

    NB: mutates ``similarities``
    """
    scoreds = []
    aeneid_freqs = aeneid_data.get_token_frequencies()
    lucan_freqs = lucan_data.get_token_frequencies()
    sqrt_aeneid_anti_freqs = np.sqrt(1 - (aeneid_freqs / aeneid_freqs.max()))
    sqrt_lucan_anti_freqs = np.sqrt(1 - (lucan_freqs / lucan_freqs.max()))
    aeneid_non_word_indices = _get_non_word_indices(aeneid_data)
    lucan_non_word_indices = _get_non_word_indices(lucan_data)
    similarities[aeneid_non_word_indices, :] = 0
    similarities[:, lucan_non_word_indices] = 0
    aeneid_bounds = _get_sentence_boundaries(aeneid_data)
    lucan_bounds = _get_sentence_boundaries(lucan_data)
    for i, (a_start_raw,
            a_end_raw) in enumerate(zip(aeneid_bounds[:-1],
                                        aeneid_bounds[1:])):
        # ignore initial [CLS] and final [SEP]
        a_start = a_start_raw + 1
        a_end = a_end_raw - 1
        for j, (l_start_raw, l_end_raw) in enumerate(
                zip(lucan_bounds[:-1], lucan_bounds[1:])):
            l_start = l_start_raw + 1
            l_end = l_end_raw - 1
            # looking at only the relevant similarities
            focus = similarities[a_start:a_end, l_start:l_end]
            # find the highest simliarities with Lucan associated with each
            # token from the Aeneid
            maxargs = focus.argmax(axis=-1)
            # calculate the relevant scaling factors
            factors = np.multiply(sqrt_aeneid_anti_freqs[a_start:a_end],
                                  sqrt_lucan_anti_freqs[l_start + maxargs])
            # apply scaling factors to highest similarities
            # https://stackoverflow.com/a/23435843
            unsummed = np.multiply(focus[np.arange(focus.shape[0]), maxargs],
                                   factors)
            score = unsummed.sum()
            # normalize by number of tokens in Aeneid passage
            comparisons_count = a_end - a_start
            scoreds.append((score / comparisons_count, i, j))
    scoreds.sort(reverse=True)
    return scoreds


def _score_by_rescaled_max_word_factor_norm(aeneid_data, lucan_data,
                                            similarities):
    """Adjust simliarities by scaled frequencies of highest simliarity word
    tokens compared; normalize by scaled frequencies

    NB: mutates ``similarities``
    """
    scoreds = []
    aeneid_freqs = aeneid_data.get_token_frequencies()
    lucan_freqs = lucan_data.get_token_frequencies()
    sqrt_aeneid_anti_freqs = np.sqrt(1 - (aeneid_freqs / aeneid_freqs.max()))
    sqrt_lucan_anti_freqs = np.sqrt(1 - (lucan_freqs / lucan_freqs.max()))
    aeneid_non_word_indices = _get_non_word_indices(aeneid_data)
    lucan_non_word_indices = _get_non_word_indices(lucan_data)
    similarities[aeneid_non_word_indices, :] = 0
    similarities[:, lucan_non_word_indices] = 0
    aeneid_bounds = _get_sentence_boundaries(aeneid_data)
    lucan_bounds = _get_sentence_boundaries(lucan_data)
    for i, (a_start_raw,
            a_end_raw) in enumerate(zip(aeneid_bounds[:-1],
                                        aeneid_bounds[1:])):
        # ignore initial [CLS] and final [SEP]
        a_start = a_start_raw + 1
        a_end = a_end_raw - 1
        for j, (l_start_raw, l_end_raw) in enumerate(
                zip(lucan_bounds[:-1], lucan_bounds[1:])):
            l_start = l_start_raw + 1
            l_end = l_end_raw - 1
            # looking at only the relevant similarities
            focus = similarities[a_start:a_end, l_start:l_end]
            # find the highest simliarities with Lucan associated with each
            # token from the Aeneid
            maxargs = focus.argmax(axis=-1)
            # calculate the relevant scaling factors
            factors = np.multiply(sqrt_aeneid_anti_freqs[a_start:a_end],
                                  sqrt_lucan_anti_freqs[l_start + maxargs])
            # normalize scaling factors
            factors = factors / factors.sum()
            # apply scaling factors to highest similarities
            # https://stackoverflow.com/a/23435843
            unsummed = np.multiply(focus[np.arange(focus.shape[0]), maxargs],
                                   factors)
            score = unsummed.sum()
            scoreds.append((score, i, j))
    scoreds.sort(reverse=True)
    return scoreds


def _score_by_rescaled_max_diff(aeneid_data, lucan_data, similarities):
    """Adjust simliarities by scaled frequencies of highest simliarity word
    tokens compared, further scaled by difference from median simliarity

    NB: mutates ``similarities``
    """
    scoreds = []
    aeneid_non_word_indices = _get_non_word_indices(aeneid_data)
    lucan_non_word_indices = _get_non_word_indices(lucan_data)
    similarities[aeneid_non_word_indices, :] = 0
    similarities[:, lucan_non_word_indices] = 0
    aeneid_bounds = _get_sentence_boundaries(aeneid_data)
    lucan_bounds = _get_sentence_boundaries(lucan_data)
    for i, (a_start_raw,
            a_end_raw) in enumerate(zip(aeneid_bounds[:-1],
                                        aeneid_bounds[1:])):
        # ignore initial [CLS] and final [SEP]
        a_start = a_start_raw + 1
        a_end = a_end_raw - 1
        for j, (l_start_raw, l_end_raw) in enumerate(
                zip(lucan_bounds[:-1], lucan_bounds[1:])):
            l_start = l_start_raw + 1
            l_end = l_end_raw - 1
            # looking at only the relevant similarities
            focus = similarities[a_start:a_end, l_start:l_end]
            # find the highest simliarities with Lucan associated with each
            # token from the Aeneid
            maxargs = focus.argmax(axis=-1)
            maxes = focus[np.arange(focus.shape[0]), maxargs]
            # find median of similarities
            medians = np.median(focus, axis=-1)
            diffs = maxes - medians
            # calculate the relevant scaling factors
            norm_term = diffs.max()
            if norm_term == 0:
                factors = 1 / len(diffs)
            else:
                factors = diffs / norm_term
            # apply scaling factors to highest similarities
            unsummed = np.multiply(maxes, factors)
            score = unsummed.sum()
            # normalize by number of tokens in Aeneid passage
            comparisons_count = a_end - a_start
            scoreds.append((score / comparisons_count, i, j))
    scoreds.sort(reverse=True)
    return scoreds


def _score_by_threshold_and_rescaled_max_word(aeneid_data, lucan_data,
                                              similarities):
    """Adjust simliarities by scaled frequencies of highest simliarity word
    tokens compared that are in the top x%

    NB: mutates ``similarities``
    """
    scoreds = []
    aeneid_freqs = aeneid_data.get_token_frequencies()
    lucan_freqs = lucan_data.get_token_frequencies()
    sqrt_aeneid_anti_freqs = np.sqrt(1 - (aeneid_freqs / aeneid_freqs.max()))
    sqrt_lucan_anti_freqs = np.sqrt(1 - (lucan_freqs / lucan_freqs.max()))
    aeneid_non_word_indices = _get_non_word_indices(aeneid_data)
    lucan_non_word_indices = _get_non_word_indices(lucan_data)
    similarities[aeneid_non_word_indices, :] = 0
    similarities[:, lucan_non_word_indices] = 0
    # zero out bottom 90% after ignoring non-word tokens
    threshold = np.quantile(similarities.ravel(), 0.9)
    mask = similarities >= threshold
    similarities[~mask] = 0
    aeneid_bounds = _get_sentence_boundaries(aeneid_data)
    lucan_bounds = _get_sentence_boundaries(lucan_data)
    for i, (a_start_raw,
            a_end_raw) in enumerate(zip(aeneid_bounds[:-1],
                                        aeneid_bounds[1:])):
        # ignore initial [CLS] and final [SEP]
        a_start = a_start_raw + 1
        a_end = a_end_raw - 1
        for j, (l_start_raw, l_end_raw) in enumerate(
                zip(lucan_bounds[:-1], lucan_bounds[1:])):
            l_start = l_start_raw + 1
            l_end = l_end_raw - 1
            # looking at only the relevant similarities
            focus = similarities[a_start:a_end, l_start:l_end]
            # find the highest simliarities with Lucan associated with each
            # token from the Aeneid
            maxargs = focus.argmax(axis=-1)
            # calculate the relevant scaling factors
            factors = np.multiply(sqrt_aeneid_anti_freqs[a_start:a_end],
                                  sqrt_lucan_anti_freqs[l_start + maxargs])
            # apply scaling factors to highest similarities
            # https://stackoverflow.com/a/23435843
            unsummed = np.multiply(focus[np.arange(focus.shape[0]), maxargs],
                                   factors)
            score = unsummed.sum()
            # normalize by number of tokens in Aeneid passage
            comparisons_count = a_end - a_start
            scoreds.append((score / comparisons_count, i, j))
    scoreds.sort(reverse=True)
    return scoreds


def _score_with_stopwords(aeneid_data, lucan_data, similarities):
    return stop.score(aeneid_data, lucan_data, similarities)


def _score_with_stopwords_and_threshold(aeneid_data, lucan_data, similarities):
    return stop.score_thresholded(aeneid_data, lucan_data, similarities)


def _score_with_stopwords_and_lemmata(aeneid_data, lucan_data, similarities):
    return stop.score_lemmata(aeneid_data, lucan_data, similarities)


def _score_by_mean_pool_cos_sim(aeneid_data, lucan_data, _):
    scoreds = []
    aeneid_bounds = _get_sentence_boundaries(aeneid_data)
    lucan_bounds = _get_sentence_boundaries(lucan_data)
    for i, (a_start,
            a_end) in enumerate(zip(aeneid_bounds[:-1], aeneid_bounds[1:])):
        for j, (l_start,
                l_end) in enumerate(zip(lucan_bounds[:-1], lucan_bounds[1:])):
            pooled_aeneid = aeneid_data.embeddings[a_start:a_end, :].mean(
                axis=0)
            normed_pooled_aeneid = pooled_aeneid / np.linalg.norm(
                pooled_aeneid)
            pooled_lucan = lucan_data.embeddings[l_start:l_end, :].mean(axis=0)
            normed_pooled_lucan = pooled_lucan / np.linalg.norm(pooled_lucan)
            scoreds.append((np.dot(normed_pooled_aeneid,
                                   normed_pooled_lucan), i, j))
    scoreds.sort(reverse=True)
    return scoreds


def _print_scoreds(aeneid_data, lucan_data, scoreds):
    for score, i, j in scoreds[:10]:
        print(''.join(['-'] * 40))
        print(score)
        a_sent = aeneid_data.sentences[i]
        print('-', aeneid_data.lines[a_sent.line_span[0]].tag)
        print(a_sent.sentence)
        l_sent = lucan_data.sentences[j]
        print('-', lucan_data.lines[l_sent.line_span[0]].tag)
        print(l_sent.sentence)


def _get_sentence_boundaries(data):
    result = []
    for i, token in enumerate(data.tokens_iter()):
        if token.token == '[CLS]':
            result.append(i)
    result.append(i)
    return result


WORD_REGEX = re.compile(r'[a-zA-Z]+', flags=re.UNICODE)


def _get_non_word_indices(data):
    result = []
    for i, token in enumerate(data.tokens_iter()):
        if token.token.startswith('[') or not WORD_REGEX.search(token.token):
            result.append(i)
    return result


def _display_lines(neighbors, distances, k, n, source_data, target_data):
    for i in range(k):
        cur_target_token = target_data.get_token(i)
        cur_target_line = target_data.get_line_for(cur_target_token)
        print(cur_target_line.get_highlighted_with(cur_target_token))
        for j, source_index in enumerate(neighbors[i]):
            if j >= n:
                break
            cur_source_token = source_data.get_token(source_index)
            cur_source_line = source_data.get_line_for(cur_source_token)
            print('\t', f'({distances[i, j]})',
                  cur_source_line.get_highlighted_with(cur_source_token))


def _display_sentences(neighbors, similarities, k, n, source_data,
                       target_data):
    for i in range(k):
        cur_target_token = target_data.get_token(i)
        _display_top_sentences(cur_target_token, neighbors[i], similarities[i],
                               n, source_data, target_data)


def _display_top_sentences(token, token_neighbors, token_similarities, n,
                           source_data, target_data):
    print(''.join(['='] * 40))
    _print_aligned_sentence_lines(target_data, token, '')
    for j, source_index in enumerate(token_neighbors):
        if j >= n:
            break
        cur_source_token = source_data.get_token(source_index)
        print(''.join(['-'] * 40))
        _print_aligned_sentence_lines(source_data, cur_source_token,
                                      f'    ({token_similarities[j]})  ')


def _print_aligned_sentence_lines(data, token, prefix):
    line = data.get_line_for(token)
    tag = line.tag
    sentence = data.get_sentence_for(token)
    out_lines = sentence.get_highlighted_with(token).split('\n')
    for i, out_line in enumerate(out_lines):
        if sentence.line_span[0] + i == token.line_index:
            starter = prefix + tag + '  '
        else:
            starter = prefix + ''.join([' ' for _ in range(len(tag) + 2)])
        print(starter + out_line)


def _evaluate(aeneid_data, lucan_data, scoreds, parallels):
    """Evaluate scored sentences on Lucan-Vergil benchmark"""
    RANDOM_SEED = 12345
    organized = OrganizedScoreds(aeneid_data, lucan_data, scoreds)
    data = [[] for _ in range(5)]
    X = []
    y = []
    ones = []
    for parallel in parallels:
        score = organized.search(parallel)
        if score is not None:
            if not np.isnan(score):
                data[parallel.rating - 1].append(score)
            X.append(score)
            y.append(parallel.rating - 1)
            if parallel.rating == 1:
                ones.append((score, parallel))
    X = np.array(X).reshape(len(X), 1)
    y = np.array(y)
    print('# Full Logistic Regression')
    performance = cross_validate(
        make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'),
                      LogisticRegression(dual=False, class_weight='balanced')),
        X,
        y,
        cv=StratifiedKFold(shuffle=True, random_state=RANDOM_SEED),
        scoring={
            'matthews': make_scorer(matthews_corrcoef),
            'spearmanr': make_scorer(lambda x, y: spearmanr(x, y)[0])
        })
    print('Matthews', performance['test_matthews'],
          performance['test_matthews'].mean())
    print('Spearmanr', performance['test_spearmanr'],
          performance['test_spearmanr'].mean())
    print('# Full Raw Score')
    performance = cross_validate(
        make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'),
                      IdentityEstimator()),
        X,
        y,
        cv=StratifiedKFold(shuffle=True, random_state=RANDOM_SEED),
        scoring={'spearmanr': make_scorer(lambda x, y: spearmanr(x, y)[0])})
    print('Spearmanr', performance['test_spearmanr'],
          performance['test_spearmanr'].mean())
    y[y < 2] = 0
    y[y >= 2] = 1
    print('# Binary Logistic Regression')
    binary_perf = cross_validate(
        make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'),
                      LogisticRegression(dual=False, class_weight='balanced')),
        X,
        y,
        cv=StratifiedKFold(shuffle=True, random_state=RANDOM_SEED),
        scoring={
            'matthews': make_scorer(matthews_corrcoef),
            'spearmanr': make_scorer(lambda x, y: spearmanr(x, y)[0])
        })
    print('Matthews', binary_perf['test_matthews'],
          binary_perf['test_matthews'].mean())
    print('Spearmanr', binary_perf['test_spearmanr'],
          binary_perf['test_spearmanr'].mean())
    print('# Binary Raw Score')
    binary_perf = cross_validate(
        make_pipeline(SimpleImputer(missing_values=np.nan, strategy='mean'),
                      IdentityEstimator()),
        X,
        y,
        cv=StratifiedKFold(shuffle=True, random_state=RANDOM_SEED),
        scoring={'spearmanr': make_scorer(lambda x, y: spearmanr(x, y)[0])})
    print('Spearmanr', binary_perf['test_spearmanr'],
          binary_perf['test_spearmanr'].mean())
    ones.sort(key=lambda x: x[0], reverse=True)
    # for score, parallel in ones[:5]:
    # print(score)
    # print(parallel)
    return data


class IdentityEstimator(BaseEstimator):

    def __init__(self):
        pass

    def fit(self, X, y):
        X, y = check_X_y(X, y, accept_sparse=True)
        self.is_fitted_ = True
        # `fit` should always return `self`
        return self

    def predict(self, X):
        X = check_array(X, accept_sparse=True)
        check_is_fitted(self, 'is_fitted_')
        return X.copy()


def _plot(results):
    fig, axes = plt.subplots(nrows=1, ncols=len(results), sharey=True)
    for data, ax in zip(results, axes):
        ax.violinplot(data)
        labels = np.arange(1, 6)
        ax.set_xticks(labels)
        ax.set_xticklabels(labels)
    plt.show()


@dataclass
class BenchParallel:
    """Class for keeping track of a parallel in the Lucan-Vergil benchmark"""
    lucan_book: int
    lucan_line: int
    lucan_snippet: str
    aeneid_book: int
    aeneid_line: int
    aeneid_snippet: str
    rating: int


def _read_benchmark():
    parallels = []
    bench_path = Path('data/aen_luc1_hand.txt')
    with bench_path.open() as ifh:
        # ignore header line
        next(ifh)
        for line in ifh:
            data = line.strip().split('\t')
            parallels.append(
                BenchParallel(lucan_book=int(data[0]),
                              lucan_line=int(data[1]),
                              lucan_snippet=data[2][1:-1],
                              aeneid_book=int(data[3]),
                              aeneid_line=int(data[4]),
                              aeneid_snippet=data[5][1:-1],
                              rating=int(data[6])))
    return parallels


class OrganizedScoreds:

    def __init__(self, aeneid_data, lucan_data, scoreds):
        self._organized = _organize(aeneid_data, lucan_data, scoreds)
        self.aeneid_data = aeneid_data
        self.lucan_data = lucan_data

    def search(self, parallel: BenchParallel) -> Optional[float]:
        aeneid_book = parallel.aeneid_book
        if aeneid_book in self._organized:
            source_tree = self._organized[aeneid_book]
            potentials = []
            for source_interval in sorted(source_tree[parallel.aeneid_line]):
                target_dict = source_interval.data
                lucan_book = parallel.lucan_book
                if lucan_book in target_dict:
                    target_tree = target_dict[lucan_book]
                    for target_interval in sorted(
                            target_tree[parallel.lucan_line]):
                        potentials.append(target_interval.data)
            if len(potentials) == 0:
                return None
            if len(potentials) == 1:
                score, _, _ = potentials[0]
                return score
            aeneid_snippet = parallel.aeneid_snippet
            lucan_snippet = parallel.lucan_snippet
            ranked = [
                (fuzz.ratio(aeneid_snippet,
                            self.aeneid_data.sentences[i].sentence) +
                 fuzz.ratio(lucan_snippet,
                            self.lucan_data.sentences[j].sentence), score)
                for score, i, j in potentials
            ]
            ranked.sort()
            return ranked[-1][1]
        return None


def _organize(aeneid_data, lucan_data, scoreds):
    results = {}
    for score, i, j in scoreds:
        source_book, source_start, source_end = _get_locus(aeneid_data, i)
        if source_book not in results:
            results[source_book] = IntervalTree()
        source_tree = results[source_book]
        if not source_tree.overlaps(source_start, source_end):
            source_tree[source_start:source_end] = {}
        for interval in source_tree[source_start:source_end]:
            if interval.begin == source_start and interval.end == source_end:
                target_dict = interval.data
                break
        assert 'target_dict' in locals()
        target_book, target_start, target_end = _get_locus(lucan_data, j)
        if target_book not in target_dict:
            target_dict[target_book] = IntervalTree()
        target_tree = target_dict[target_book]
        target_tree[target_start:target_end] = (score, i, j)
    return results


def _get_locus(data, sent_ind):
    sent = data.sentences[sent_ind]
    start, end = sent.line_span
    start_tag = data.lines[start].tag
    if end < len(data.lines):
        end_tag = data.lines[end].tag
    else:
        end_tag = data.lines[-1].tag
    start_book, start_line = _parse_tag(start_tag)
    end_book, end_line = _parse_tag(end_tag)
    if start_book != end_book:
        # one past last line is in next book
        _, last_line = _parse_tag(data.lines[end - 1].tag)
        end_line = last_line + 1
    if start_line != end_line:
        return start_book, start_line, end_line
    return start_book, start_line, start_line + 1


def _parse_tag(tag):
    locus = tag.strip().split()[-1]
    book, line = locus.split('.')
    return int(book), int(line)


if __name__ == '__main__':
    _main()
