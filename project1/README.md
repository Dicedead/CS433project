# ML Higgs Project, CS-433 Project I

Repository for the team `PythonIsTrash`.

## Reproducibility

### Assumptions
The ``data`` folder should contain ``train.csv`` as well as ``test.csv``.

### Reproducing the test results
Run the file ``run.py``. It will output a ``csv`` file of the desired format as ``data/submissions/submission_final.csv``.

## Folder organization
### root
The required files ``implementations.py``, ``run.py`` and ``README.md`` are at root level.
### sources
Contains all the code that was not mandatory.
* ``preprocessing.py`` encapsulates all the preprocessing steps described in the project report.
* ``metrics.py`` contains functions to compute the accuracy, the precision, the recall, etc.
* ``validation.py`` encloses cross validation functions.
* ``train.py`` runs cross validations and trains the model, saving weight vectors to ``data/weights.txt``.
* ``test.py`` outputs ``csv`` submissions.
* ``additional_implementations.py`` has multiple implementations of logistic regression and its variants.
### grading_tests
Contains the publicly given tests.
