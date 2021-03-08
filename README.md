### Instalation

Open a docker container with the following command:

    sudo docker run --gpus all --privileged -it -v ${PWD}:/working_dir -w /working_dir tensorflow/tensorflow:1.12.0-devel-gpu-py3

Execute the following command to compile the custom ops of tensorflow:

    cd IEProtLib/tf_ops
    python genCompileScript.py --cudaFolder /usr/local/cuda
    sh compile.sh

We already provide a compiled version of the library for the docker container, so if you are using the docker container indicated above you can skip the compilation.


### Download the preprocessed datasets.

In the following links the different datasets can be downloaded:

Enzymes vs Non-Enzymes: 
    https://drive.google.com/uc?export=download&id=1KTs5cUYhG60C6WagFp4Pg8xeMgvbLfhB
    Extract content in: Datasets/data/ProteinsDD/
Scope 1.75:
    https://drive.google.com/uc?export=download&id=1chZAkaZlEBaOcjHQ3OUOdiKZqIn36qar
    Extract content in: Datasets/data/HomologyTAPE/
Protein function:
    https://drive.google.com/uc?export=download&id=1udP6_90WYkwkvL1LwqIAzf9ibegBJ8rI
    Extract content in: Datasets/data/ProtFunct

### Train Ennzymes vs Non-Enzymes

Execute the following commands to train a network on the task:

    cd Tasks/ProteinsDD
    python Train.py --configFile confs/train_fold0.ini
    python Train.py --configFile confs/train_fold1.ini
    python Train.py --configFile confs/train_fold2.ini
    python Train.py --configFile confs/train_fold3.ini
    python Train.py --configFile confs/train_fold4.ini
    python Train.py --configFile confs/train_fold5.ini
    python Train.py --configFile confs/train_fold6.ini
    python Train.py --configFile confs/train_fold7.ini
    python Train.py --configFile confs/train_fold8.ini
    python Train.py --configFile confs/train_fold9.ini

### Train SCOPe 1.75

Execute the following commands to train a network on the task:

    cd Tasks/ProtHomology
    python Train.py --configFile confs/train.ini

To evalute the trained model on the different test set use the following commands:

    python Test.py --configFile confs/test_fold.ini
    python Test.py --configFile confs/test_superfamily.ini
    python Test.py --configFile confs/test_family.ini

### Train Protein function prediction

Execute the following commands to train a network on the task:

    cd Tasks/ProtFunct
    python Train.py --configFile confs/train.ini

To evaluate the trained model execute:

    python Test.py --configFile confs/test.ini