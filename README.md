# pefe-ief

A `pefe-system` module for running models'
Inference on Extracted Features (IEF).

The feature vectors are extracted to a LMDB
file by [`pefe-loader`](https://github.com/pefe-system/pefe-loader)
and [`pefe-agent`(s)](https://github.com/pefe-system/pefe-agent)

- [pefe-ief](#pefe-ief)
  - [Usage](#usage)
    - [1. Integrate pefe-ief into your model's project](#1-integrate-pefe-ief-into-your-models-project)
      - [1a. Wrap your model](#1a-wrap-your-model)
      - [1b. Run pefe-ief](#1b-run-pefe-ief)
    - [2. View the results using pefe-ief-viz](#2-view-the-results-using-pefe-ief-viz)

## Usage

### 1. Integrate pefe-ief into your model's project

#### 1a. Wrap your model

Wrap your model into a class.

An example:

```python
from pefe_ief.models.abstract_model import AbstractModel
import lightgbm as lgb

class SOREL20M_LGBM(AbstractModel):
    def do_load(self, model_path):
        """Load the model from model_path. Initialize your stuff."""
        self._model = lgb.Booster(model_file=model_path)

    def do_extract_features(self, bytes):
        """Extract features from file content (bytes - a bytearray).
        Return one feature vector (i.e. 1D np.array)"""
        feature_vector = do_something_here()
        return feature_vector
    
    def do_predict(self, feature_vectors):
        """Given a list of input feature vectors (i.e. 2D np.array),
        return the inferred probabilities corresponding to each
        input vector respectively (i.e. 1D np.array)"""
        y_probs = self._model.predict(feature_vectors)
        return y_probs
    
    def do_get_batch_size(self):
        """How many records do you wish us to load into RAM/VRAM
        and process at once?"""
        return 65536
```

#### 1b. Run pefe-ief

Provided that you have wrapped two models,
`SOREL20M_LGBM` (as seen in the above example)
and `SOREL20M_FFNN`. To run IEF on these
models:

```python
from pefe_ief import IEF
from pefe_ief.models.abstract_model import AbstractModel
from pefe_ief.dataset import PEFELMDBDataset

# 0/6. Import your wrapped models
from somewhere_in_your_project import SOREL20M_LGBM, SOREL20M_FFNN

# 1/6. Specify a directory to contain IEF results
ief = IEF("/your/results/dir")

# 2/6. Specify the directories that contain
#      the model files e.g. checkpoints, weights etc.
#      Here, two directories containing model checkpoints
#      of two different model types. For each eligible
#      file in a directory, the corresponding model
#      class e.g. SOREL20M_LGBM will be instantiated
#      and your do_load() method will be called with
#      the file's absolute path as argument.
models_dirs_and_classes = {
    "/dir/containing/lgbm/model/files": SOREL20M_LGBM,
    "/dir/containing/ffnn/model/files": SOREL20M_FFNN,
}

# 3/6. However, for each file recursively found in those
#      directories, this function will be called to
#      determine whether that file is indeed a model
#      checkpoint file. An example of how it would be
#      called:
#
#      for each file in those directories, recursive iterations:
#           file_path = path of the file
#           if not is_model_checkpoint_file(SOREL20M_FFNN, file_path):
#               continue # skip file
#           else:
#               model = model_class()
#               model.load(file_path)
#               # ...
#
#      The function shall return True if the given
#      file_path is indeed a model checkpoint file,
#      and False otherwise.
def is_model_checkpoint_file(model_class, file_path):
    # type: (Type[AbstractModel], str) -> str
    return file_path.endswith(".pt") or file_path.endswith(".model") 

# 4/6. To identify models in the generated reports and
#      visualizations, this function would be called to
#      get the models' names displayed in those, given
#      the model_class, checkpoint_path (same as 3/6)
#      and also model_type_name, which is actually just
#      `model_class.__name__`.
def get_model_checkpoint_name(model_class, model_type_name, checkpoint_path):
    # type: (Type[AbstractModel], str, str) -> str | None
    import os
    return os.path.splitext(os.path.basename(checkpoint_path))[0] # simple implementation: just get the file name

# 5/6. Load your test set. You could read X_test and
#      y_test from LMDB using pefe-ief's PEFELMDBDataset
#      like this, or you could read from anywhere
#      else. X_test shall be a 2D ndarray (m rows, n cols)
#      while Y_test shall be a 1D ndarray (m rows), where m
#      is the number of test samples, n is the number of
#      features aka dimensionality of the input feature vector.
X_test, y_test = PEFELMDBDataset().read("/path/to/lmdb/directory")

# 6/6. Finally, run pefe-ief. The results will be generated in
#      the directory you specified at 1/6.
ief.run(
    models_dirs_and_classes=models_dirs_and_classes,
    is_model_checkpoint_file=is_model_checkpoint_file,
    get_model_checkpoint_name=get_model_checkpoint_name,
    X_test=X_test, y_test=y_test,
)
```

### 2. View the results using pefe-ief-viz

Use [`pefe-ief-viz`](https://github.com/pefe-system/pefe-ief-viz)
to view the generated results.
