# Meta Learning for POS Tagging 
Implementation of three Meta Learning approaches : [MAML](https://arxiv.org/pdf/1703.03400.pdf), [Reptile and FOMAML](https://arxiv.org/pdf/1803.02999.pdf) for POS Tagging in PyTorch.

Datset has been borrowed from [Universal Dependencies](https://universaldependencies.org/)

Two languages are used to train the model Hindi and Marathi.

## Training the Model
This would also print the test results

to train and test the model run
```
python train.py
```
Optional Arguments are: 
```
--learning_rate The learning rate for MAML
--epsilon       The value of epsilon for updating model parameters in Reptile
```
