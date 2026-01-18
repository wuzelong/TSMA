# TSMA

## usage

1. Environmental preparation

To install the required Python dependencies, run:

```
pip install -r requirements.txt
```

For reproducibility, we also packaged the conda environment. You can [[download]](https://drive.google.com/file/d/1ZCknTbkJqgMasXHrp-hycKx5zJqWwZJi/view?usp=sharing) it and run `tar -xzf TSMA.tar.gz -C ~/.conda/envs/`.

> The experiments are conducted using **Python 3.9** and ​**CUDA 12.8**​. Other versions may be compatible but are not officially tested.

2. Download Datasets

Please download and extract the data into `./dataset`.

* Supervised training
  Datasets from [TSLib](https://github.com/thuml/Time-Series-Library) : [[Download]](https://cloud.tsinghua.edu.cn/f/4d83223ad71047e28aec/).
* Large-scale pre-training
  * [ERA5-Family](https://www.ecmwf.int/en/forecasts/dataset/ecmwf-reanalysis-v5) form [OPenLTM](https://github.com/thuml/OpenLTM) : [[Download]](https://cloud.tsinghua.edu.cn/f/7fe0b95032c64d39bc4a/).
  * [UTSD](https://huggingface.co/datasets/thuml/UTSD) form [OPenLTM](https://github.com/thuml/OpenLTM): [[Download]](https://cloud.tsinghua.edu.cn/f/93868e3a9fb144fe9719/).

3. training
   
   run

```
python run.py --config config/short-term/ETTh1.yaml
```

