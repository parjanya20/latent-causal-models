# Causal Discovery For Latent Causal Models
Code for ICLR 2025 paper - Differentiable Causal Discovery For Latent Hierarchical Causal Models (https://arxiv.org/abs/2411.19556)

## Requirements

* Python >= 3.8
* PyTorch >= 1.9.0
* NumPy >= 1.19.0
* tqdm >= 4.64.0


## Project Structure
The code is organized into three main modules:

1. **algorithm/**: Core implementation
2. **data/**: Example data and ground truth adjacency matrices as .npy files
3. **evaluation/**: Evaluation metrics

~~~
├── algorithm/
│   ├── init.py              
│   ├── get_causal_graph.py      # file with get_graph() function,it returns the loss and causal graph
│   ├── hierarchical_model.py    # hierarchical model class
│   └── utils.py                 # Helper functions and utilities
├── data/
│   ├── init.py              
│   ├── adj_matrix_tree.npy      # Adjacency matrix data for example tree structure
│   ├── adj_matrix_v_structure.npy # Adjacency matrix for example v-structure
│   ├── tree.npy                 # Example tree structure dataset
│   └── v_structure.npy          # Example v-structure
├── evaluation/
│   ├── init.py              
│   └── metrics.py               # file with function to get SHD and F1
└── Example.ipynb                # Jupyter notebook with usage example
~~~



## Usage

### Quick Start
```python
import numpy as np
from algorithm.get_causal_graph import get_graph

# Load your data
X = np.array(...)  # shape: (n_samples, n_features)

# Get predicted causal graph
loss, adj_matrix_pred = get_graph(X)
```

### Example
A complete example using tree structure data is provided in `Example.ipynb`.

**Note**: Since the optimization is non-convex, it is recommended to run the code multiple times and select the graph with the minimum objective function value (loss).

## Citation
If you find our work useful, please cite our paper:

```bibtex
@article{prashant2024differentiable,
 title={Differentiable Causal Discovery For Latent Hierarchical Causal Models},
 author={Prashant, Parjanya and Ng, Ignavier and Zhang, Kun and Huang, Biwei},
 journal={arXiv preprint arXiv:2411.19556},
 year={2024}
}
```
