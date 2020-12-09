# HDD-Net: Hybrid Detector Descriptor with Mutual Interactive Learning
Code for the ACCV20 paper:

```text
"HDD-Net: Hybrid Detector Descriptor with Mutual Interactive Learning".
Axel Barroso-Laguna, Yannick Verdie, Benjamin Busam, Krystian Mikolajczyk. ACCV 2020.
```
[[Paper on arxiv](https://arxiv.org/abs/2005.05777)]

## Prerequisite

Python 3.7 is required for running HDD-Net code. Use Conda to install the dependencies:

```bash
conda create --name hddnet_environment tensorflow-gpu=1.15.0
conda activate hddnet_environment 
conda install scikit-image
conda install -c conda-forge opencv
conda install -c conda-forge scikit-image
```

## Feature Extraction

`extract_multiscale_features.py` can be used to extract HDD-Net features for a given list of images. The list of images must contain the full path to them, if they do not exist, an error will raise. 

The script generates two numpy files, one '.kpt' for keypoints, and a '.dsc' for descriptors. The output format of the keypoints is as follow:

- `keypoints` [`N x 4`] array containing the positions of keypoints `x, y`, scales `s` and their scores `sc`. 


Arguments:

  * list-images: File containing the image paths for extracting features.
  * results-dir: The output path to save the extracted features.
  * checkpoint-dir: The path to the checkpoint file to load the model weights. Default: Pretrained HDD-Net.
  * num-points: The number of desired features to extract. Default: 2048.
  * extract-MS: Set to True if you want to extract multi-scale features. Default: True.


Run the following script to generate the keypoint and descriptor numpy files from the image allocated in `im_test` directory. 

```bash
python extract_multiscale_features.py --list-images im_test/im_path.txt --results-dir im_test/
```

## BibTeX

If you use this code in your research, please consider citing our paper:

```bibtex
@InProceedings{Barroso-Laguna2020ACCV,
    author = {Barroso-Laguna, Axel and Verdie, Yannick and Busam, Benjamin and Mikolajczyk, Krystian},
    title = {{HDD-Net: Hybrid Detector Descriptor with Mutual Interactive Learning}},
    booktitle = {Proceedings of the 2020 IEEE/CVF Asian Conference on Computer Vision},
    year = {2020},
}
