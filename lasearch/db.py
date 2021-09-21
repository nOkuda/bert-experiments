"""Database for storing and searching embeddings"""
import hashlib
import pickle
from pathlib import Path

from lasearch.latinbert import LatinBERT
from lasearch.reader import read_tessfile
from lasearch.tokenize import LatinSentenceTokenizer


class FlatFileDatabase:
    """Uses pickled data to store information

    Attributes
    ----------
    db_dir : Path
        Directory where flat files are stored
    """

    def __init__(self, db_dir: Path):
        self.db_dir = db_dir.expanduser().resolve()
        self.db_dir.mkdir(parents=True, exist_ok=True)
        self._sent_toker = LatinSentenceTokenizer()

    def ingest(self,
               tessfilepath: Path,
               bert_model: LatinBERT,
               overwrite: bool = False):
        """Add the given .tess file to the database

        By default, the database will not overwrite a flat file that already
        exists
        """
        tessfilepath = tessfilepath.expanduser().resolve()
        flatfilepath = self._get_flat_file_path(tessfilepath)
        if not overwrite and flatfilepath.exists():
            raise ValueError(
                f'Database already has information for {str(tessfilepath)}')
        data = read_tessfile(tessfilepath, self._sent_toker, bert_model)
        with flatfilepath.open(mode='wb') as ofh:
            pickle.dump(data, ofh)

    def retrieve(self, tessfilepath: Path):
        """Retrieve stored data for given .tess file

        Raises ValueError if database doesn't have .tess file ingested
        """
        tessfilepath = tessfilepath.expanduser().resolve()
        flatfilepath = self._get_flat_file_path(tessfilepath)
        if not flatfilepath.exists():
            raise ValueError(
                f'Database has no information for {str(tessfilepath)}')
        with flatfilepath.open(mode='rb') as ifh:
            return pickle.load(ifh)

    def _get_flat_file_path(self, filepath: Path) -> Path:
        """Generate the flat file path associated with ``filepath``"""
        return self.db_dir / hashlib.sha256(bytes(filepath)).hexdigest()
