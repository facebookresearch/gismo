# InverseCooking 2.0

Code for inversecooking2.0: merges image-to-set prediction with previous inversecooking and adds functionalities to move towards multi-modal generation.

## Installation

This code uses Python 3.8.5 (Anaconda), PyTorch 1.6 and cuda version 10.1.

- Installing pytorch:
```bash
$ conda install pytorch torchvision cudatoolkit=10.1 -c pytorch
```

- Install dependencies
```bash
$ pip install -r requirements.txt --upgrade
```

- Additional dependencies for developers

```bash
$ pip install -r requirements_dev.txt --upgrade
```

## Datasets

#### Recipe1M

- Download [Recipe1M](http://im2recipe.csail.mit.edu/dataset/download) (registration required) and extract under ```/path/to/recipe1m/```.
- The contents of ```/path/to/recipe1m/``` should be the following:
```
det_ingrs.json
layer1.json
layer2.json
images/
images/train
images/val
images/test
```
- Pre-process dataset and build vocabularies with:

```bash
$ python src/utils/recipe1m_utils.py --recipe1m_path path_to_recipe1m
```
Resulting files will be stored under ```/path/to/recipe1m/preprocessed```.
- Fill in ```configs/datapaths.json``` with the path to recipe1m dataset: ````"recipe1m": "/path/to/recipe1m/"````

## Training

*Note: all python calls below must be run from* `./src`.

Checkpoints will be saved under a directory ```"<save_dir>/<dataset>/<model_name>/<image_model>/<experiment_name>/"```,  specified by ```--save_dir```, ```--dataset```, ```--model_name```, ```--image_model``` and ```--experiment_name```.

The recommended way to train the models reported in the paper is to use the JSON configuration files provided in
```configs``` folder. We have provided one configuration file for each combination of dataset, set predictor (model_name) and image backbone (image_model). The naming convention is ```configs/dataset/image_model_model_name.json```.

Training can be run as in the following example command:
```bash
$ python train.py --save_dir ../checkpoints --resume --seed SEED --dataset DATASET \
--image_model IMAGE_MODEL --model_name MODEL_NAME --use_json_config
```
where DATASET is a dataset name (e.g. `recipe1m`), IMAGE_MODEL and MODEL_NAME are among the models listed above (e.g. `resnet50` and `ff_bce_cat`) and SEED is the value of a random seed (e.g. `1235`).

Check training progress with Tensorboard from ```../checkpoints```:
```bash
$ tensorboard --logdir='.' --port=6006
```

## Evaluation

*Note: all python calls below must be run from* `./src`.

Calculate evaluation metrics as in the following example command:
```bash
$ python eval.py --eval_split test --models_path PATH --dataset DATASET --batch_size 100
```
where DATASET is a dataset name (e.g. `recipe1m`) and PATH is the path to the saved models folder.

## Pre-trained models
TBD

## Contributing

### Developer tools

The following commands should be used prior to submitting a fix:

- `make format` to format all the files with and remove useless imports
- `make test` to ensure that all unit tests are green
- `make check` to run basic static checks on the codebase

## License

TBD