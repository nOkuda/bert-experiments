import time
from pathlib import Path

from lasearch.db import FlatFileDatabase
from lasearch.latinbert import LatinBERT


def _main():
    bert_model = LatinBERT(
        tokenizerPath=
        'latin-bert/models/subword_tokenizer_latin/latin.subword.encoder',
        bertPath='best_model')
    db = FlatFileDatabase(db_dir=Path('flatdb_finetuned'))
    latin_tess_files_dir = Path('data')
    aeneid_path = latin_tess_files_dir / 'vergil.aeneid.tess'
    lucan1_path = latin_tess_files_dir / 'lucan.bellum_civile.part.1.tess'
    _timed_ingest(db, aeneid_path, bert_model, 'Aeneid ingest time:')
    _timed_ingest(db, lucan1_path, bert_model, 'Lucan 1 ingest times:')


def _timed_ingest(db, tessfile_path, bert_model, message):
    start = time.time()
    db.ingest(tessfile_path, bert_model)
    ingest_time = time.time() - start
    print(message, ingest_time)


if __name__ == '__main__':
    _main()
