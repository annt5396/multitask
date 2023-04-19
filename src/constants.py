import os
from datasets import Sequence, Value, Features
from datasets import Dataset, DatasetDict


EXAMPLE_FEATURES = Features(
    {
        "guid": Value(dtype="string", id=None),
        "question": Value(dtype="string", id=None),
        "context": Value(dtype="string", id=None),
        "answers": Sequence(
            feature={
                "text": Value(dtype="string", id=None),
                "answer_start": Value(dtype="int32", id=None),
            },
        ),
        "is_impossible": Value(dtype="bool", id=None),
        "title": Value(dtype="string", id=None),
        "classtype": Value(dtype="string", id=None),
        "source": Value(dtype="string", id=None),
        "dataset": Value(dtype="string", id=None),
    }
)

QUESTION_COLUMN_NAME = "question"
CONTEXT_COLUMN_NAME = "context"
ANSWER_COLUMN_NAME = "answers"
ANSWERABLE_COLUMN_NAME = "is_impossible"
ID_COLUMN_NAME = "guid"

DEFAULT_CONFIG_FILE = os.path.join(
    os.path.realpath(__file__), "args/default_config.yaml"
)