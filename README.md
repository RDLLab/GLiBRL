# Meta-RL with Bayesian Linear Task Models (GLiBRL)

GLiBRL combines learned neural features with Bayesian linear reward and transition models. Closed-form Bayesian updates infer the task from new interactions. A belief-conditioned policy uses this inference to adapt its behaviour.

This is the official implementation of Meta-RL with Bayesian Linear Task Models (GLiBRL), NeurIPS 2026. The implementation is adapted from [MetaWorld+](https://github.com/rainx0r/metaworld-algorithms).

**Authors:** Jingyang You and Hanna Kurniawati.

**Paper:** [arXiv](https://arxiv.org/abs/2512.20974) | [PDF](https://arxiv.org/pdf/2512.20974)



## Code Structure

- **examples/**: This directory contains the scripts that launch experiments.
- **metaworld_algorithms/**: This is the core of the repository, containing the building blocks for experiments.
    - **config/**: Defines the configuration dataclasses for all components of an experiment (e.g., environment, algorithm, networks).
    - **envs/**: Contains code for creating and managing environments, particularly for the Meta-World benchmark.
    - **nn/**: Provides neural network building blocks and architectures.
    - **rl/**: Contains the reinforcement learning algorithms.
        - **algorithms/**: This is where the implementations of the RL algorithms like MAML-TRPO, RL2, and SAC reside.
    - **run.py**: This file contains the main Run class that orchestrates the entire experiment lifecycle, from initialization to training and evaluation.

Typical workflow:

1.  An experiment is defined in a script within the **examples/** directory by composing various configuration objects.
2. This script then passes the configuration to the **Run** class in metaworld_algorithms/run.py.
3. The **Run** class uses a factory pattern to create the appropriate environment and algorithm based on the provided configuration.
4. Finally, the **Run** class executes the training loop of the selected algorithm.

## Installation

### From a clone of the repository

0. Install [uv](https://docs.astral.sh/uv/)
1. Create a virtual environment: `uv venv .venv --python 3.12`
2. Activate the virtual environment: `source .venv/bin/activate`
3. Install jax-cuda: `uv pip install -e ".[cuda12]"`

## Reproducing the results

Seeds can be set by an extra --seed argument. We enable float64 in GLiBRL in MetaWorld for better numerical stability with longer horizons. 

- `python examples/meta_learning/glibrlsac_antdir.py` 
- `python examples/meta_learning/glibrlsac_halfcheetahdir.py` 
- `python examples/meta_learning/glibrlsac_halfcheetahvel.py` 
- `JAX_ENABLE_X64=1 python examples/meta_learning/glibrl_ml10.py` 
- `JAX_ENABLE_X64=1 python examples/meta_learning/glibrl_ml45.py` 

GLiBRL without noise inference can be run by setting the property `full_bayesian` to `False`, in `glibrl_ml10.py` or `glibrl_ml45.py`

## Citation

If you use GLiBRL in your research, please cite:

```bibtex
@inproceedings{you2026glibrl,
  title     = {Meta-{RL} with Bayesian Linear Task Models},
  author    = {You, Jingyang and Kurniawati, Hanna},
  booktitle = {Advances in Neural Information Processing Systems},
  year      = {2026},
  url       = {https://arxiv.org/abs/2512.20974}
}
```
