import json
import re
import unicodedata
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np
from fuzzywuzzy import fuzz
from intervaltree import IntervalTree
from scipy.sparse import csr_matrix


def score_lemmata(aeneid_data, lucan_data, similarities):
    scoreds = []

    aeneid_organized = Intervalled(aeneid_data)
    lucan_organized = Intervalled(lucan_data)
    aeneid_zeroing_indices = _get_zeroing_indices(aeneid_data)
    lucan_zeroing_indices = _get_zeroing_indices(lucan_data)
    similarities[aeneid_zeroing_indices, :] = 0
    similarities[:, lucan_zeroing_indices] = 0

    # find indices where lemmata overlap
    matched = _get_matched_positions(aeneid_data, lucan_data,
                                     similarities.shape)
    # zero-out unmatched similarities
    similarities[~matched] = 0

    aeneid_freqs = aeneid_data.get_token_frequencies()
    lucan_freqs = lucan_data.get_token_frequencies()
    sqrt_aeneid_anti_freqs = np.sqrt(1 - (aeneid_freqs / aeneid_freqs.max()))
    sqrt_lucan_anti_freqs = np.sqrt(1 - (lucan_freqs / lucan_freqs.max()))

    for parallel in _read_benchmark():
        aeneid_found = aeneid_organized.search(parallel.aeneid_book,
                                               parallel.aeneid_line,
                                               parallel.aeneid_snippet)
        lucan_found = lucan_organized.search(parallel.lucan_book,
                                             parallel.lucan_line,
                                             parallel.lucan_snippet)
        if aeneid_found is None:
            print('Could not find Aeneid:', parallel.aeneid_book,
                  parallel.aeneid_line, parallel.aeneid_snippet)
        if lucan_found is None:
            print('Could not find Lucan:', parallel.lucan_book,
                  parallel.lucan_line, parallel.lucan_snippet)
        if aeneid_found is not None and lucan_found is not None:
            # ignore initial [CLS] and final [SEP]
            a_start = aeneid_found[0] + 1
            a_end = aeneid_found[1] - 1
            l_start = lucan_found[0] + 1
            l_end = lucan_found[1] - 1
            # looking at only the relevant similarities
            focus = similarities[a_start:a_end, l_start:l_end]
            # abort early if there aren't enough similarity scores
            comparisons_count = np.sum(focus.sum(axis=-1) != 0)
            if comparisons_count < 1:
                scoreds.append(
                    (np.nan,
                     aeneid_data.get_token(aeneid_found[0]).sentence_index,
                     lucan_data.get_token(lucan_found[0]).sentence_index))
                continue
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
            raw_score = unsummed.sum()
            # normalize by number of non-zero tokens in Aeneid passage
            score = raw_score / comparisons_count
            # found = FoundParallel(aeneid_found=aeneid_found,
            # lucan_found=lucan_found,
            # score=score,
            # rating=parallel.rating)
            scoreds.append(
                (score, aeneid_data.get_token(aeneid_found[0]).sentence_index,
                 lucan_data.get_token(lucan_found[0]).sentence_index))
    scoreds.sort(reverse=True)
    return scoreds


def _get_matched_positions(aeneid_data, lucan_data, final_shape):
    lemma_indexer = LemmaIndexer()
    a_row_inds, a_col_inds = _get_row_inds_and_col_inds(
        lemma_indexer, aeneid_data)
    l_row_inds, l_col_inds = _get_row_inds_and_col_inds(
        lemma_indexer, lucan_data)
    lemma_dim = max(a_col_inds.max(), l_col_inds.max()) + 1
    a_csr = csr_matrix(
        (np.ones(len(a_row_inds), dtype=np.bool), (a_row_inds, a_col_inds)),
        shape=(final_shape[0], lemma_dim))
    l_csr = csr_matrix(
        (np.ones(len(l_row_inds), dtype=np.bool), (l_row_inds, l_col_inds)),
        shape=(final_shape[1], lemma_dim))
    return a_csr.dot(l_csr.transpose()).toarray()


class LemmaIndexer:

    def __init__(self):
        self.lookup = _load_latin_lemmas()
        self.encountereds = dict()

    def get_lemma_indices(self, unlemmatized: str) -> Iterable[int]:
        founds = self.lookup.get(_preprocess(unlemmatized), [])
        for f in founds:
            if f not in self.encountereds:
                self.encountereds[f] = len(self.encountereds)
        return [self.encountereds[f] for f in founds]


def _load_latin_lemmas():
    results = {}
    lex_path = Path('data') / 'la.lexicon.csv'
    with lex_path.open(encoding='utf-8') as ifh:
        for line in ifh:
            line = line.strip()
            if line:
                word, _, headword = line.split(',')
                word = _preprocess(word)
                headword = _preprocess(headword)
                if word not in results:
                    results[word] = []
                results[word].append(headword)
    return results


def _preprocess(token):
    normalized = unicodedata.normalize('NFKD', token).lower()
    j_replaced = re.sub('j', 'i', normalized)
    v_replaced = re.sub('v', 'u', j_replaced)
    return re.sub(r'[^a-zA-Z]+', '', v_replaced)


def _get_row_inds_and_col_inds(lemma_indexer, data):
    row_ind = []
    col_ind = []
    for i, token in enumerate(data.tokens_iter()):
        for lemma_index in lemma_indexer.get_lemma_indices(token.token):
            row_ind.append(i)
            col_ind.append(lemma_index)
    return np.array(row_ind), np.array(col_ind)


def score_thresholded(aeneid_data, lucan_data, similarities):
    scoreds = []

    aeneid_organized = Intervalled(aeneid_data)
    lucan_organized = Intervalled(lucan_data)
    aeneid_zeroing_indices = _get_zeroing_indices(aeneid_data)
    lucan_zeroing_indices = _get_zeroing_indices(lucan_data)
    similarities[aeneid_zeroing_indices, :] = 0
    similarities[:, lucan_zeroing_indices] = 0

    threshold = np.quantile(similarities.ravel(), 0.99)
    drop = similarities < threshold
    print(np.sum(similarities[drop]) / drop.sum())
    similarities[drop] = 0

    aeneid_freqs = aeneid_data.get_token_frequencies()
    lucan_freqs = lucan_data.get_token_frequencies()
    sqrt_aeneid_anti_freqs = np.sqrt(1 - (aeneid_freqs / aeneid_freqs.max()))
    sqrt_lucan_anti_freqs = np.sqrt(1 - (lucan_freqs / lucan_freqs.max()))

    for parallel in _read_benchmark():
        aeneid_found = aeneid_organized.search(parallel.aeneid_book,
                                               parallel.aeneid_line,
                                               parallel.aeneid_snippet)
        lucan_found = lucan_organized.search(parallel.lucan_book,
                                             parallel.lucan_line,
                                             parallel.lucan_snippet)
        if aeneid_found is None:
            print('Could not find Aeneid:', parallel.aeneid_book,
                  parallel.aeneid_line, parallel.aeneid_snippet)
        if lucan_found is None:
            print('Could not find Lucan:', parallel.lucan_book,
                  parallel.lucan_line, parallel.lucan_snippet)
        if aeneid_found is not None and lucan_found is not None:
            # ignore initial [CLS] and final [SEP]
            a_start = aeneid_found[0] + 1
            a_end = aeneid_found[1] - 1
            l_start = lucan_found[0] + 1
            l_end = lucan_found[1] - 1
            # looking at only the relevant similarities
            focus = similarities[a_start:a_end, l_start:l_end]
            # abort early if there aren't enough similarity scores
            comparisons_count = np.sum(focus.sum(axis=-1) != 0)
            if comparisons_count < 2:
                continue
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
            raw_score = unsummed.sum()
            # normalize by number of non-zero tokens in Aeneid passage
            score = raw_score / comparisons_count
            # found = FoundParallel(aeneid_found=aeneid_found,
            # lucan_found=lucan_found,
            # score=score,
            # rating=parallel.rating)
            scoreds.append(
                (score, aeneid_data.get_token(aeneid_found[0]).sentence_index,
                 lucan_data.get_token(lucan_found[0]).sentence_index))
    scoreds.sort(reverse=True)
    return scoreds


def score(aeneid_data, lucan_data, similarities):
    scoreds = []

    aeneid_organized = Intervalled(aeneid_data)
    lucan_organized = Intervalled(lucan_data)
    aeneid_zeroing_indices = _get_zeroing_indices(aeneid_data)
    lucan_zeroing_indices = _get_zeroing_indices(lucan_data)
    similarities[aeneid_zeroing_indices, :] = 0
    similarities[:, lucan_zeroing_indices] = 0

    aeneid_freqs = aeneid_data.get_token_frequencies()
    lucan_freqs = lucan_data.get_token_frequencies()
    sqrt_aeneid_anti_freqs = np.sqrt(1 - (aeneid_freqs / aeneid_freqs.max()))
    sqrt_lucan_anti_freqs = np.sqrt(1 - (lucan_freqs / lucan_freqs.max()))

    for parallel in _read_benchmark():
        aeneid_found = aeneid_organized.search(parallel.aeneid_book,
                                               parallel.aeneid_line,
                                               parallel.aeneid_snippet)
        lucan_found = lucan_organized.search(parallel.lucan_book,
                                             parallel.lucan_line,
                                             parallel.lucan_snippet)
        if aeneid_found is None:
            print('Could not find Aeneid:', parallel.aeneid_book,
                  parallel.aeneid_line, parallel.aeneid_snippet)
        if lucan_found is None:
            print('Could not find Lucan:', parallel.lucan_book,
                  parallel.lucan_line, parallel.lucan_snippet)
        if aeneid_found is not None and lucan_found is not None:
            # ignore initial [CLS] and final [SEP]
            a_start = aeneid_found[0] + 1
            a_end = aeneid_found[1] - 1
            l_start = lucan_found[0] + 1
            l_end = lucan_found[1] - 1
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
            raw_score = unsummed.sum()
            # normalize by number of non-zero tokens in Aeneid passage
            comparisons_count = np.sum(focus.sum(axis=-1) != 0)
            score = raw_score / comparisons_count
            # found = FoundParallel(aeneid_found=aeneid_found,
            # lucan_found=lucan_found,
            # score=score,
            # rating=parallel.rating)
            scoreds.append(
                (score, aeneid_data.get_token(aeneid_found[0]).sentence_index,
                 lucan_data.get_token(lucan_found[0]).sentence_index))
    scoreds.sort(reverse=True)
    return scoreds


def _get_sentence_boundaries(data):
    result = []
    for i, token in enumerate(data.tokens_iter()):
        if token.token == '[CLS]':
            result.append(i)
    result.append(i)
    return result


WORD_REGEX = re.compile(r'[a-zA-Z]+', flags=re.UNICODE)


def _get_zeroing_indices(data):
    result = []
    for i, token in enumerate(data.tokens_iter()):
        token_str = token.token
        if _is_non_word(token_str) or _is_stopword(token_str):
            result.append(i)
    return result


def _is_non_word(token_str):
    return token_str.startswith('[') or not WORD_REGEX.search(token_str)


# https://github.com/aurelberra/stopwords/blob/master/stopwords_latin.json
def _get_stopwords():
    result = []
    jsonfilepath = Path('data') / 'stopwords_latin.json'
    if not jsonfilepath.exists():
        raise Exception('Could not find ' + str(jsonfilepath) +
                        ' (try downloading from ' +
                        'https://github.com/aurelberra/stopwords/' +
                        'blob/master/stopwords_latin.json)')
    with open(str(jsonfilepath)) as ifh:
        full_stopwords = json.load(ifh)
    result.extend(full_stopwords['CONJUNCTIONS'])
    result.extend(full_stopwords['PREPOSITIONS'])
    result.extend(full_stopwords['ADVERBS'])
    for _, more_words in full_stopwords['PRONOUNS'].items():
        result.extend(more_words)
    result.extend(full_stopwords['VERBS']['sum'])
    result.extend(['-que', '-ve', '-ue'])
    return result


STOPWORDS = _get_stopwords()


def _is_stopword(token_str):
    return token_str in STOPWORDS


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


class Intervalled:
    """Representation of locus intervals for sentences in a text"""

    def __init__(self, tess_data):
        self.tess_data = tess_data
        self._organized = _build_intervals(tess_data)

    def search(self, book: int, line: int,
               snippet: str) -> Optional[Tuple[int, int]]:
        """Given a book, line number, and text snippet, find the index range of
        tokens pertaining to that locus

        If the locus could not be found, returns None
        """
        if book in self._organized:
            tree = self._organized[book]
            potentials = []
            for interval in sorted(tree):
                potentials.extend(interval.data)
            if len(potentials) == 0:
                return None
            if len(potentials) == 1:
                return potentials[0]
            ranked = [(fuzz.ratio(
                snippet,
                self.tess_data.get_sentence_for(self.tess_data.get_token(
                    a[0])).sentence), a) for a in potentials]
            ranked.sort()
            return ranked[-1][1]
        return None


def _build_intervals(tess_data):
    results = {}
    cur_sent_ind = 0
    first_token_ind = 0
    for i, token in enumerate(tess_data.tokens_iter()):
        if token.sentence_index != cur_sent_ind:
            _add_interval(results, tess_data, first_token_ind, i)
            # update bookkeeping
            cur_sent_ind = token.sentence_index
            first_token_ind = i
    # account for last interval
    _add_interval(results, tess_data, first_token_ind, i)
    return results


def _add_interval(results, tess_data, first_token_ind, end_token_ind):
    """Mutates results by adding a new interval"""
    # end token is one past last token
    book, start, end = _get_loci(tess_data, first_token_ind, end_token_ind - 1)
    if book not in results:
        results[book] = IntervalTree()
    tree = results[book]
    if not tree.overlaps(start, end):
        tree[start:end] = []
    for interval in tree[start:end]:
        if interval.begin == start and interval.end == end:
            relevant = interval.data
            break
    if 'relevant' not in locals():
        tree[start:end] = [(first_token_ind, end_token_ind)]
    else:
        relevant.append((first_token_ind, end_token_ind))


def _get_loci(tess_data, first_token_ind, last_token_ind):
    first_token = tess_data.get_token(first_token_ind)
    start_tag = tess_data.get_line_for(first_token).tag
    last_token = tess_data.get_token(last_token_ind)
    last_tag = tess_data.get_line_for(last_token).tag
    start_book, start_line = _parse_tag(start_tag)
    last_book, last_line = _parse_tag(last_tag)
    # end line is one past the last line
    return start_book, start_line, last_line + 1


def _parse_tag(tag):
    locus = tag.strip().split()[-1]
    book, line = locus.split('.')
    return int(book), int(line)


@dataclass
class FoundParallel:
    aeneid_found: Tuple[int, int]
    lucan_found: Tuple[int, int]
    score: float
    rating: int
