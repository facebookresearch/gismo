# InverseCooking 2.0

Code for inversecooking2.0: merges image-to-set prediction with previous inversecooking and adds functionalities to move towards multi-modal generation.

<br>

## Installation

This code uses Python 3.8.5 (Anaconda), PyTorch 1.7, Torchvision 0.8.1 and CUDA version 10.1.

- Installing pytorch:

```bash
conda install -c pytorch pytorch=1.7.1 torchvision cudatoolkit=10.2
conda install -c conda-forge -c fvcore -c iopath fvcore iopath
conda install pytorch3d -c pytorch3d
```

- Install dependencies

```bash
pip install -r requirements.txt --upgrade
```

- Additional dependencies for developers (optional)

```bash
pip install -r requirements_dev.txt --upgrade
pip install -e ".[dev]"
```

- Install NLTK punkt (for pre-processing of datasets)

```bash
python -c "import nltk; nltk.download('punkt')"
python3 -m spacy download en_core_web_lg
```

- Verify that the install worked

```bash
python -c "import spacy; spacy.load('en_core_web_lg')"
```

<br>

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

- Link the dataset to your current folder (the other option is to modify "path" in the configuration of the dataset)

```
ln -s /path/to/recipe1m/ data/recipe1m
```

- Pre-process the dataset with:

```
python preprocess.py dataset=recipe1m
```

<br>

## Training

### Running experiments

Training can be run as in the following example command:

    python train.py task=im2recipe name=im2recipe

This command will look for the definition of the experiment "im2recipe" in the configuration
file "conf/experiments/im2recipe.yaml" and run this experiment.

Evaluation can be run as in the following example command:

    python eval.py task=im2recipe name=im2recipe

### Running on SLURM

Running on SLURM requires only to add the SLURM configuration to the command line:

    python train.py task=im2recipe name=im2recipe slurm=<SLURM_CONF>

Existing SLURM configurations can be found in the folder `conf/slurm/<SLURM_CONF>.yaml`.
Feel free to add configurations for your specific cluster.
You can find an example with `conf/slurm/devlab.yaml`.

### Monitoring progress

Check training progress with Tensorboard from the folder in which the checkpoint are saved:

    tensorboard --logdir='.' --port=6006

<br>

## Reproducing experiments

To reproduce the experiments in inversecooking1.0, train the image-to-ingredient model as follows on 2 gpus:

    python train.py task=im2ingr name=im2ingr_resnet50_ff_bce_cat

Once trained, update the im2recipe.yaml file with the following entry (or edit the existing one):
```
 im2recipe_invcooking1.0:
   comment: 'inverse cooking 1.0 model'
   parent: im2recipe
   pretrained_im2ingr:
     freeze: True
     load_pretrained_from: /path/to/im2ingr-im2ingr_resnet50_ff_bce_cat/best.ckpt
```
Then, train the image-to-recipe model as follows on 1 node and 8 gpus:

    python train.py task=im2recipe name=im2recipe_invcooking1.0

Finally, you can evaluate your model as follows:

    python eval.py task=im2recipe name=im2recipe_invcooking1.0 eval_checkpoint_dir=/directory/of/the/checkpoint

Note that models will be evaluated on the val_all data split of Recipe1M without using teacher forcing by default. If you would like to change the evaluation set, please change the flag eval_split under the recipe1m config. Possible eval_split choices are: train, val (subset of 5k samples from val_all), val_all, and test. If you would like to use teacher forcing in the evaluation, please set the flag ingr_teachforce.test to True (see the im2recipe.yaml file).

## Evaluation

TBD

<br>

## Pre-trained models

TBD

<br>

## Contributing

### Developer tools

The following commands should be used prior to submitting a fix:

- `make format` to format all the files with and remove useless imports
- `make test` to ensure that all unit tests are green
- `make check` to run basic static checks on the codebase

<br>

## License

TBD
