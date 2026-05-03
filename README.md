
## Requirements

Recommended environment:

- Python 3.8+
- PyTorch with CUDA support
- NumPy
- SciPy
- Matplotlib
- timm

Install common Python dependencies with:

```bash
pip install numpy scipy matplotlib timm
```

Please install the appropriate PyTorch version separately according to your CUDA environment.

## Project Structure

```text
DSTG-Net
|-- checkpoint/
|-- image/
|   |-- arc.PNG
|-- model/
|   |-- dstg_net.py
|   |-- modules/
|-- utils/
|-- main_h36m_3d.py
|-- main_cmu_3d.py
|-- main_3dpw_3d.py
```

## Data Preparation

Download the datasets and organize them according to the formats expected by this codebase.

### Human3.6M

`--data_dir` should point to the directory that directly contains the subject folders:

```text
[dataset path]
|-- S1
|-- S5
|-- S6
|-- S7
|-- S8
|-- S9
|-- S11
```

The loader reads files in the form:

```text
S1/walking_1.txt
S1/walking_2.txt
...
```

### CMU-MoCap

`--data_dir` should point to the directory that contains `train/` and `test/`:

```text
[dataset path]
|-- train
|   |-- basketball
|   |   |-- basketball_1.txt
|   |   |-- basketball_2.txt
|   |-- basketball_signal
|   |-- directing_traffic
|   |-- jumping
|   |-- running
|   |-- soccer
|   |-- walking
|   |-- washwindow
|-- test
|   |-- basketball
|   |-- basketball_signal
|   |-- directing_traffic
|   |-- jumping
|   |-- running
|   |-- soccer
|   |-- walking
|   |-- washwindow
```

### 3DPW
`--data_dir` should point to:

```text
[dataset path]
|-- train
|   |-- *.pkl
|-- validation
|   |-- *.pkl
|-- test
|   |-- *.pkl
```

## Training

### Train on Human3.6M

```bash
python main_h36m_3d.py --data_dir [dataset path] --input_n 10 --output_n 10 --kernel_size 10 --dct_n 20 --batch_size 32 --test_batch_size 64 --in_features 66 --cuda_idx cuda:0 --d_model 16 --lr_now 0.0005 --epoch 50 --test_sample_num -1
```


## Evaluation

Add `--is_eval` to the corresponding training command.

Example:

```bash
python main_h36m_3d.py --data_dir [dataset path] --input_n 10 --output_n 10 --kernel_size 10 --dct_n 20 --test_batch_size 64 --in_features 66 --cuda_idx cuda:0 --d_model 16 --lr_now 0.0005 --test_sample_num -1 --is_eval
```

To resume training from the last checkpoint, add:

```bash
--is_load
```
Our pre-training parameters cannot be uploaded due to size limitations.

## Acknowledgments

This codebase is built on top of **PGBIG**.
