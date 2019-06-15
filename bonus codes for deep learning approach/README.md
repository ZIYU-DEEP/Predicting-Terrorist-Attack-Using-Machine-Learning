# Spatial-Time RNN

This theory is proposed by the paper of [Predicting the Next Location: A Recurrent Model with Spatial and Temporal Contexts](https://pdfs.semanticscholar.org/5bdf/0970034d0bb8a218c06ba3f2ddf97d29103d.pdf). The code is adapted from [this repo](https://github.com/yongqyu/STRNN) which uses Gowalla dataset, and we modified the data loading, preprocessing and training phase to allow the code suitable for the GTD data.

The deep learning approach is currently only a bonus to our project and we will contiuning updating this repo for better performance. Our current working direction is improving the calculation of linear interpolation and the loss function calculation.

## Requirements
- Python 2.7 +
- Pytorch 0.4.0 + with GPU

## Dataset

The original GTD dataset is not contained in this dataset, please refer to the `../data` folder for the full data. We transfer the full data in the form of `gtd.csv`. You can refer to the notebook `gtddata_manipulation.ipynb`, which is a simple sample of how we accomplish the transfer.

## Usage

### 0. `csv_txt.py`

In this file, we simply tansfer the `gtd.csv` into `gtd.txt`, and filtered out rows with unknown attributes. You can skip this part as we already provided the processed `gtd.txt` in the directory.

### 1. `data_loader.py`

This could be regard as a file containing helper functions. There are two major functions in this file, one is `load_data`, which will later on be used in `preprocess.py`, and in turn generate a preliminary `train_file` (70%), `test_file` (20%) and `validation file` (10%). 

The next function is `treat_prepro`, which will later on be used in the `train_torch.py`. Basically, it extract the attributes (e.g. group ID, location ID, distance, etc.) of data processed by `preprocess.py`.

### 2. `preprocess.py`

As we have actually prepared the GTD data in the repository, so you could simply run the following line:

```bash
$ python preprocess.py
```

### 3. `train_torch.py`

After running the previous code, you could then run the following command:

```bash
$ python train_torch.py
```

We notice that on one of our computers, this file will report an unclear Pytorch error message during the half of the training phase. We are trying to figure out the problem and update with a more compatible code.