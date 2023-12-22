# Entropy-Vergence Metric

The following is a full implementation of the Entropy (con/di)-Vergence Metric as described in Rosen and Dale 2023. We've taken strides to make this package as easy to implement as possible.

At its core there are two things that researchers need to define for the most basic version of the EVM algorithm. Those are:

1. A word vector model (we suggest those implemented in the HuggingFace transformer library)
2. An entropy object

Demonstrating how this would look in code:

```python
from EVM import EVM, languageModel
from transformers import AutoTokenizer, AutoModelForSequenceClassification

wv = languageModel(
    vector_model = AutoModelForSequenceClassification.from_pretrained('roberta-base'),
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
)

mod = EVM(
    word_vector_model = wv
)

mod('I hate mondays with the sort of burning passion reserved for the spiteful', 'There is no worse day than wednesday during the week'), \
mod('I hate mondays with the sort of burning passion reserved for the spiteful', 'I adore the feeling of a good monday meeting')
```

The above returns the entropy for the first sentence upon having read the second sentence, consistent with the equations and formulation listed in Rosen and Dale 2023.