# bert-experiments

Code to run Latin BERT experiments as reported in my dissertation.

## Installing

  1. Create and use a virtual environment first.
  2. Install the version of PyTorch that will work for your hardware: <https://pytorch.org/get-started/locally/>. Experiments were done on 1.9.
  3. Install the other dependencies
     ```
     pip install -r requirements.txt
     ```
  4. Install CLTK Latin tokenizer models
     ```
     python3 -c "from cltk.corpus.utils.importer import CorpusImporter; corpus_importer = CorpusImporter('latin'); corpus_importer.import_corpus('latin_models_cltk')"
     ```
  5. Download Latin BERT model, Latin SBERT model, subtokenizer, and .tess files
     ```
     ./download.sh
     ```

## Running

```
python3 ingest.py # ingest Lucan and Aeneid
python3 full_compute.py # score passages against each other

# with finetuned model
python3 ingest_finetuned.py
python3 full_compute_finetuned.py
```

## Notes

LatinBERT is the work of David Bamman and Patrick J. Burns.
Their preprint paper is available on arXiv: <https://arxiv.org/abs/2009.10053>.
The code is available at <https://github.com/dbamman/latin-bert>.
