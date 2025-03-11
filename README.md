# Scalable Out-of-distribution Robustness in the Presence of Unobserved Confounders
Code for Scalable Out-of-distribution Robustness in the Presence of Unobserved Confounders (AISTATS 2025) (https://arxiv.org/abs/2411.19923)

## Project Structure
The code is organized into two main modules:

1. **src/**: Core implementation
2. **data/**: Example data 

## Usage

### Quick Start
```python
import numpy as np
import tensorflow as tf
from src import main

# Load your data
X = np.array(...)  # shape: (n_samples, n_features)
Y = np.array(...)  # shape: (n_samples,)
s = np.array(...)  # shape: (n_samples, n_s)
z = np.array(...)  # shape: (n_samples, n_z)
t = np.array(...)  # shape: (n_samples,) - the training or test indicator.
# Get predictions and accuracy
predictions, accuracy = main.run(X, y, s, z, t)
```

### Example
A complete example using tree structure data is provided in `Example.ipynb`.


## Citation
If you find our work useful, please cite our paper:

```bibtex
@article{prashant2024scalable,
  title={Scalable Out-of-distribution Robustness in the Presence of Unobserved Confounders},
  author={Prashant, Parjanya and Khatami, Seyedeh Baharan and Ribeiro, Bruno and Salimi, Babak},
  journal={arXiv preprint arXiv:2411.19923},
  year={2024}
}

```
