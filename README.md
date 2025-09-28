# NowcastNet - PyTorch Implementation

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python Version](https://img.shields.io/badge/python-3.9%2B-blue)](https://www.python.org/)
[![PyTorch Version](https://img.shields.io/badge/pytorch-2.8.0%2B-red)](https://pytorch.org/)

This project is a PyTorch implementation of NowcastNet, based on the paper "Skilful nowcasting of extreme precipitation with NowcastNet" by the Tsinghua University team. The model is constructed as a GAN architecture that includes an evolution network for simulating physical processes, in addition to the standard generator and discriminator components.

## Model Overview

NowcastNet is an advanced deep learning model for precipitation nowcasting that combines physical modeling with generative adversarial networks. The model consists of three main components:

- **Evolution Network**: Simulates realistic physical processes
- **Generator**: Generates fake data based on data features to fool the discriminator
- **Discriminator**: Distinguishes between real and generated outputs

This implementation faithfully reproduces the architecture described in the original paper, providing accurate short-term precipitation forecasts.

## Requirements

- **Python**: 3.9+
- **PyTorch**: 2.8.0+
- **GPU**: NVIDIA RTX 3090 (or compatible CUDA device)
- **Memory**: Recommended 16GB or more
- **Storage**: Sufficient space for NJU-CPOL dataset and model files

## Installation

Clone this repository:
```bash
git clone https://github.com/Hitlia/NowcastNet.git
cd NowcastNet
```

## Dataset

The model is trained and tested using the **NJU-CPOL dual-polarization radar dataset** provided by Nanjing University. This dataset contains high-quality radar observations suitable for precipitation nowcasting tasks.

## Project Structure

```
NowcastNet/
├── run_all.py              # Main training script
├── test.py                 # Testing script
├── params.py               # Model parameter configuration
├── make_model.py           # Main model construction
├── blocks.py               # Building blocks for the model
├── submodels.py            # Sub-modules implementation
├── utils.py                # Utility functions and basic operations
├── dataset.py              # DataLoader construction
├── loss_function/          # Loss functions for GAN components
│   ├── evolution_loss.py   # Evolution network loss
│   ├── generator_loss.py   # Generator loss
│   └── discriminator_loss.py # Discriminator loss
└── README.md
```

## Usage

### Training
To train the NowcastNet model:
```bash
python run_all.py
```

### Testing
To evaluate the trained model:
```bash
python test.py
```

## Model Architecture

### Components

1. **Evolution Network**
   - Simulates physical precipitation evolution processes
   - Captures temporal dynamics and physical constraints

2. **Generator**
   - Generates realistic precipitation fields
   - Learns to produce outputs that match the statistical properties of real data

3. **Discriminator**
   - Distinguishes between real radar observations and generated outputs
   - Provides adversarial feedback to improve generator performance

### Training Procedure

The model components are optimized sequentially during training:
1. **Evolution Network** loss computation and optimization
2. **Generator** loss computation and optimization  
3. **Discriminator** loss computation and optimization

This sequential training strategy ensures stable convergence and effective learning of all components.

## Configuration

Model parameters and training settings are configured in `params.py`, including:
- Network architecture parameters
- Training hyperparameters
- Data preprocessing settings
- Loss function weights

## Key Features

- **Physical-informed Learning**: Evolution network incorporates physical constraints
- **Adversarial Training**: GAN framework for realistic generation
- **Extreme Precipitation Focus**: Specialized for forecasting heavy rainfall events
- **Dual-polarization Radar Data**: Utilizes advanced radar observations
- **Stable Training**: Sequential optimization of network components

## Performance

The implementation aims to reproduce the skillful nowcasting performance demonstrated in the original paper, particularly for:
- Short-term precipitation forecasting (0-6 hours)
- Extreme precipitation events
- Spatial and temporal patterns of rainfall

## Citation

If you use this implementation, please cite the original paper:

```bibtex
@article{nowcastnet2023,
  title={Skilful nowcasting of extreme precipitation with NowcastNet},
  author={Yuchen Zhang, Mingsheng Long, Kaiyuan Chen, Lanxiang Xing, Ronghua Jin, Michael I. Jordan and Jianmin Wang},
  journal={Nature},
  year={2023},
  publisher={Nature Publishing Group}
}
```

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.


## Acknowledgments

- Thanks to the Tsinghua University team for the original NowcastNet paper
- Thanks to Nanjing University for providing the NJU-CPOL dual-polarization radar dataset
- Thanks to all contributors and researchers in the precipitation nowcasting community

---
