# CoTuning
Official implementation for NeurIPS 2020 paper [Co-Tuning for Transfer Learning](https://proceedings.neurips.cc//paper/2020/file/c8067ad1937f728f51288b3eb986afaa-Paper.pdf).

[News] 2021/01/13 The COCO 70 dataset used in the paper is available for download! 

# COCO 70 dataset

COCO 70 dataset is a **large-scale** classification dataset (1000 images per class) created from COCO. It is used to explore the effect of fine-tuning with a large amount of data. Check our paper if you are interested in how it was created. Please respect the original license of COCO when you use it.

To download COCO 70, follow these steps:

1. download separate files [here](https://cloud.tsinghua.edu.cn/d/b364038fd4544530bc08/) (the file is too large to upload, so I have to split it into chunks)
2. merge separate files into a single file by ```cat COCO70_splita* > COCO70.tar```

3. extract the dataset from the file by ```tar -xf COCO70.tar ```

The directory architecture looks like the following:

   ├── classes.txt #per class name per name

   ├── dev

   ├── dev.txt # [filename, class_index] per line, 0 <= class_index <= 69

   ├── test

   ├── test.txt

   ├── train

   └── train.txt 

There are 100 images per class for validation (dev.txt) and test (test.txt) respectively, and 800 images per class for training (train.txt).

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
If you use our code or use the constructed COCO-70 dataset, please consider citing:
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
If you have any problem about our code, feel free to contact ykc20@mails.tsinghua.edu.cn or kz19@mails.tsinghua.edu.cn.
