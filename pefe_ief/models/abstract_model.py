import numpy as np
from numpy import ndarray
import time
from tqdm import tqdm
import gc

class AbstractModel:
    def __init__(self):
        pass

    def do_load(self, model_path):
        # type: (AbstractModel, str) -> None
        raise NotImplementedError
    def do_extract_features(self, bytes):
        # type: (AbstractModel, bytes) -> ndarray
        raise NotImplementedError
    def do_predict(self, feature_vectors):
        # type: (AbstractModel, ndarray) -> None
        raise NotImplementedError
    def do_get_batch_size(self):
        # type: (AbstractModel) -> int
        raise NotImplementedError
    



    def load(self, model_path):
        # type: (AbstractModel, str) -> None
        return self.do_load(model_path)

    def extract_features(self, bytes):
        # type: (AbstractModel, bytes) -> ndarray
        return self.do_extract_features(bytes)
    
    def extract_features_from_single_file(self, pe_file_path):
        # type: (AbstractModel, str) -> ndarray
        with open(pe_file_path, "rb") as f:
            raw_bytes = f.read()
        
        return self.extract_features(raw_bytes)
    
    def predict(self, X_test):
        # type: (AbstractModel, ndarray) -> ndarray
        BATCH_SIZE = self.get_batch_size()
        X_test_batches = [X_test[i:i+BATCH_SIZE] for i in range(0, len(X_test), BATCH_SIZE)]
        y_probs = []
        TOTAL = 0
        for X_test_batch in tqdm(X_test_batches):
            TOTAL += len(X_test_batch)
            y_probs_batch = self.do_predict(X_test_batch)
            y_probs.extend(y_probs_batch)
        assert TOTAL == len(X_test)
        y_probs = np.array(y_probs)

        del X_test_batches
        gc.collect()
        return y_probs

    def get_batch_size(self):
        # type: (AbstractModel) -> int
        return self.do_get_batch_size()
    
    def predict_single_file(self, pe_file_path):
        # type: (AbstractModel, str) -> float

        extract_features_start_time = time.perf_counter()
        feature_vector = self.extract_features_from_single_file(pe_file_path)
        extract_features_end_time = time.perf_counter()
        extract_features_time = extract_features_end_time - extract_features_start_time

        feature_vectors = np.array([feature_vector])

        inference_start_time = time.perf_counter()
        prob = self.predict(feature_vectors)[0]
        inference_end_time = time.perf_counter()
        inference_time = inference_end_time - inference_start_time

        print(f"... feature extraction took {extract_features_time:.6f}s ; inference {inference_time:.6f}s ...")

        return prob
