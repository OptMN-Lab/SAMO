# SAMO
Official implementation of "SAMO: A Lightweight Sharpness-Aware Approach for Multi-Task Optimization with Joint Global-Local Perturbation" [ICCV 2025]


![SAMO](/misc/samo.png)

### Setup Environment

First, create the virtual environment:
```
conda create -n mtl python=3.9.7
conda activate mtl
python -m pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113
```

Then, install the repo:
```
git clone https://github.com/OptMN-Lab/SAMO.git
cd SAMO
python -m pip install -e .
```

### Run Experiment
The dataset by default should be put under `experiments/EXP_NAME/dataset/` folder where `EXP_NAME` is chosen from `{celeba, cityscapes, nyuv2, quantum_chemistry}`. To run the experiment:
```
cd experiments/EXP_NAME
sh run.sh
```

### Acknowledgements
This codebase is built on [Nash-MTL](https://github.com/AvivNavon/nash-mtl), [FAMO](https://github.com/Cranial-XIX/FAMO), [LibMTL](https://github.com/median-research-group/LibMTL), [MeZO-SVRG](https://github.com/amazon-science/mezo_svrg). We sincerely thank the authors for their efforts and contributions.
