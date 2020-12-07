# CoTuning
Original implementation for NeurIPS 2020 paper [Co-Tuning for Transfer Learning](https://proceedings.neurips.cc//paper/2020/file/c8067ad1937f728f51288b3eb986afaa-Paper.pdf).

## Dependencies
* python3
* torch == 1.1.0 (with suitable CUDA and CuDNN version)
* torchvision == 0.3.0
* scikit-learn
* numpy
* argparse
* tqdm

## Datasets
| Dataset | Download Link |
| -- | -- |
| CUB-200-2011 | http://www.vision.caltech.edu/visipedia/CUB-200-2011.html |
| Stanford Cars | http://ai.stanford.edu/~jkrause/cars/car_dataset.html |
| FGVC Aircraft | http://www.robots.ox.ac.uk/~vgg/data/fgvc-aircraft/ |

## Quick Start
```
python --gpu [gpu_num] --data_path /path/to/dataset --class_num [class_num] --trade_off 2.3 train.py 
```

## Citation
If you use this code for your research, please consider citing:
```
@article{you2020co,
  title={Co-Tuning for Transfer Learning},
  author={You, Kaichao and Kou, Zhi and Long, Mingsheng and Wang, Jianmin},
  journal={Advances in Neural Information Processing Systems},
  volume={33},
  year={2020}
}
```

## Contact
If you have any problem about our code, feel free to contact kz19@mails.tsinghua.edu.com.
