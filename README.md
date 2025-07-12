# ESC-50: Dataset for Environmental Sound Classification

The ESC-50 dataset is a labeled collection of 2000 environmental audio recordings suitable for benchmarking methods of environmental sound classification.

The dataset consists of 5-second-long recordings organized into 50 semantical classes (with 40 examples per class) loosely arranged into 5 major categories:

| <sub>Animals</sub> | <sub>Natural soundscapes & water sounds </sub> | <sub>Human, non-speech sounds</sub> | <sub>Interior/domestic sounds</sub> | <sub>Exterior/urban noises</sub> |
| :--- | :--- | :--- | :--- | :--- |
| <sub>Dog</sub> | <sub>Rain</sub> | <sub>Crying baby</sub> | <sub>Door knock</sub> | <sub>Helicopter</sub> |
| <sub>Rooster</sub> | <sub>Sea waves</sub> | <sub>Sneezing</sub> | <sub>Mouse click</sub> | <sub>Chainsaw</sub> |
| <sub>Pig</sub> | <sub>Crackling fire</sub> | <sub>Clapping</sub> | <sub>Keyboard typing</sub> | <sub>Siren</sub> |
| <sub>Cow</sub> | <sub>Crickets</sub> | <sub>Breathing</sub> | <sub>Door, wood creaks</sub> | <sub>Car horn</sub> |
| <sub>Frog</sub> | <sub>Chirping birds</sub> | <sub>Coughing</sub> | <sub>Can opening</sub> | <sub>Engine</sub> |
| <sub>Cat</sub> | <sub>Water drops</sub> | <sub>Footsteps</sub> | <sub>Washing machine</sub> | <sub>Train</sub> |
| <sub>Hen</sub> | <sub>Wind</sub> | <sub>Laughing</sub> | <sub>Vacuum cleaner</sub> | <sub>Church bells</sub> |
| <sub>Insects (flying)</sub> | <sub>Pouring water</sub> | <sub>Brushing teeth</sub> | <sub>Clock alarm</sub> | <sub>Airplane</sub> |
| <sub>Sheep</sub> | <sub>Toilet flush</sub> | <sub>Snoring</sub> | <sub>Clock tick</sub> | <sub>Fireworks</sub> |
| <sub>Crow</sub> | <sub>Thunderstorm</sub> | <sub>Drinking, sipping</sub> | <sub>Glass breaking</sub> | <sub>Hand saw</sub> |

## Download

The dataset is available for download at:

**https://github.com/karolpiczak/ESC-50**

## Results

Some results obtained with this dataset:

| <sub>Method</sub> | <sub>Accuracy</sub> |
| :--- | :--- |
| <sub>Human performance</sub> | <sub>**81.3%**</sub> |
| <sub>SVM (linear)</sub> | <sub>39.6%</sub> |
| <sub>SVM (RBF)</sub> | <sub>43.8%</sub> |
| <sub>Random Forest</sub> | <sub>44.3%</sub> |
| <sub>K-NN</sub> | <sub>62.7%</sub> |
| <sub>ConvNet (custom)</sub> | <sub>64.5%</sub> |
| <sub>ConvNet (transfer)</sub> | <sub>73.7%</sub> |

## Repository Content

This repository contains:

- **model.py**: Custom CNN architecture with residual blocks for audio classification
- **main.py**: Audio processing utilities and model inference
- **train_fixed.py**: Training script for the CNN model
- **results_fixed/**: Training results and model checkpoints
- **models/**: Saved model weights

## License

The dataset is available under the terms of the Creative Commons Attribution Non-Commercial license.

## Citing

If you find this dataset useful, please cite:

*Karol J. Piczak. ESC: Dataset for Environmental Sound Classification. Proceedings of the 23rd Annual ACM Conference on Multimedia, Brisbane, Australia, 2015.*

```
@inproceedings{piczak2015dataset,
  title={ESC: Dataset for environmental sound classification},
  author={Piczak, Karol J},
  booktitle={Proceedings of the 23rd ACM international conference on Multimedia},
  pages={1015--1018},
  year={2015},
  organization={ACM}
}
```