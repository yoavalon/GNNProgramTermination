# GNNProgramTermination
Using Graph Neural Networks for Program Termination

![Architecture](imgs/system.png)

```
https://doi.org/10.5281/zenodo.7083445
```

Classifier for program termination based on Graph Attention layers. Published in ESEC/FSE 2022 - The 30th ACM Joint European Software Engineering Conference and Symposium on the Foundations of Software Engineering.
https://arxiv.org/abs/2207.14648
To cite: 
```
@inproceedings{Alon2022UsingGN,
  title={Using Graph Neural Networks for Program Termination},
  author={Yoav Alon and Cristina David},
  year={2022}
}
```

## Run locally

#### Setup virtual environment

If not installed, install python virtual environment : 
```
pip install virtualenv 
```

Create virtual environment with directory gnnEnv
```
python3 -m venv gnnEnv
```

activate virtual environment
```
source gnnEnv/bin/activate
```

load required dependencies to virtual environment
```
python3 -m pip install -r requirements.txt
```

#### Training
```
python3 train.py
```

#### Live evaluation with Tensorboard
```
tensorboard --logdir=runs
```


#### Evaluate pretrained models 
```
python3 test.py
```

#### Evaluate test set with Tensorboard
```
tensorboard --logdir=tests
```