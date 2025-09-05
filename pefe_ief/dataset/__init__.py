import lmdb
from tqdm import trange
import msgpack
import msgpack_numpy
import gc
import numpy as np
from numpy import ndarray

msgpack_numpy.patch()

class logger:
    @classmethod
    def printl(cls, message):
        print(f"PEFEDatasetReader: {message}", flush=True)

    @classmethod
    def info(cls, message):
        return cls.printl("info: " + message)

class PEFELMDBDataset:
    def read(self, lmdb_path):
        # type: (PEFELMDBDataset, str) -> tuple[ndarray, ndarray]
        """Returns X, y"""
        logger.info(f"Loading data from {lmdb_path}")
        db = lmdb.open(lmdb_path, readonly=True, lock=False, map_size=1024 * 1024 * 1024 * 1024) # 1 TB
        y_test = []
        X_test = []
        num_entries = 0
        with db.begin() as txn:
            stat_info = txn.stat()
            num_entries = stat_info['entries']

            cursor = txn.cursor()
            PROGRESS = trange(num_entries)
            for k, v in cursor:
                PROGRESS.update()
                payload = msgpack.unpackb(v, raw=False)

                label = payload['lb']       # int
                y_test.append(label)

                features = payload['ef']    # numpy array
                X_test.append(features)

        assert PROGRESS.n == num_entries
        PROGRESS.close()

        logger.info(f"Loading data: Realigning...")
        gc.collect()
        X_test = np.array(X_test)
        gc.collect()
        y_test = np.array(y_test)
        gc.collect()

        return X_test, y_test
