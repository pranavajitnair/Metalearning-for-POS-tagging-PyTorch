# Meta Learning for POS Tagging 
Implementation of three Meta Learning approaches : [MAML](https://arxiv.org/pdf/1703.03400.pdf), [Reptile](https://arxiv.org/pdf/1803.02999.pdf) for POS Tagging in PyTorch.

Datset has been borrowed from [Universal Dependencies](https://universaldependencies.org/)

Nine languages are used to train the model.

## Training the Model
This would also print the test results

to train and test the model run
```
python train.py
```
Optional Arguments are: 
```
--learning_rate          The learning rate for MAML
--epsilon                The value of epsilon for updating model parameters in Reptile
--epochs                 Number of epochs
--K_shot_learning        How many sentences to sample for learning i.e 1-shot, 5-shot etc
--N_way_learning         How many tasks to test on
--hidden_size            Hidden size for LSTM units
--training_mode          MAML, Reptile which one to use for training the model
--inner_gradient_update  Number of inner gradient updates to perform
```
