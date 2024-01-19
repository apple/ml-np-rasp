# RASP-L in NumPy

The code in this repository accompanies the research paper, [What Algorithms can Transformers Learn? A Study in Length Generalization](https://arxiv.org/abs/2310.16028). It is provided for reference and research purposes only.

We include the NumPy-based implementation of RASP-L, as well as several of the code listings from the above paper.
We also suggest taking a look at [raskell](https://github.com/charlesfrye/raskell), an independent implementation of RASP-L in Haskell.

## Installation

Clone the repo, then install with:
```
pip install -e .
```

## Usage
To import all RASP-L core and library functions:
```
from np_rasp import *
```

For example, see [add.py](examples/add.py) for RASP-L programs for forward and reverse addition.
You can run several addition tests via:
```
cd examples
python test_add.py
```

### Citation

```
@misc{zhou2023algorithms,
      title={What Algorithms can Transformers Learn? A Study in Length Generalization}, 
      author={Hattie Zhou and Arwen Bradley and Etai Littwin and Noam Razin and Omid Saremi and Josh Susskind and Samy Bengio and Preetum Nakkiran},
      booktitle={The Twelfth International Conference on Learning Representations},
      year={2024},
      url={https://arxiv.org/abs/2310.16028}
}
```