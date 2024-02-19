# Entropy-conVergence Metric

The following is a full implementation of the convergence-entropy measurement framework (here referred to as Entropy-conVergence Metric, or EVM) as described in Rosen and Dale 2023. We've taken strides to make this package as easy to implement as possible.

At its core there are two things that researchers need to define for the most basic version of the EVM algorithm. Those are:

1. A word vector model (we suggest those implemented in the HuggingFace transformer library)
2. An entropy object

Demonstrating how this would look in code:

```python
from EVM import EVM, languageModel
from transformers import AutoTokenizer, AutoModel

wv = languageModel(
    vector_model = AutoModel.from_pretrained('roberta-base'),
    tokenizer = AutoTokenizer.from_pretrained('roberta-base')
)

mod = EVM(
    word_vector_model = wv
)

mod('I hate mondays with the sort of burning passion reserved for the spiteful', 'There is no worse day than wednesday during the week'), \
mod('I hate mondays with the sort of burning passion reserved for the spiteful', 'I adore the feeling of a good monday meeting')
```

The above returns the entropy for the first sentence upon having read the second sentence, consistent with the equations and formulation listed in Rosen and Dale 2023.

Any use of this package should cite it using the following:

```
@article{rosen_berts_2023,
	title = {{BERTs} of a feather: {Studying} inter- and intra-group communication via information theory and language models},
	shorttitle = {{BERTs} of a feather},
	url = {https://link.springer.com/10.3758/s13428-023-02267-2},
	doi = {10.3758/s13428-023-02267-2},
	journal = {Behavior Research Methods},
	author = {Rosen, Zachary P and Dale, Rick},
	year = {2023},
}
```
