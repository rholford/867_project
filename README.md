# Generate Prune Select - CISC 867 Tensorflow Reproduction

## Requirements
To install requirements:
```setup
pip install tensorflow version==2.6.0
pip install tensorflow-addons
pip install numpy
pip install scipy
pip install -r orig_requirements.txt
```
The GloVe embeddings can be downloaded from https://nlp.stanford.edu/projects/glove/, the 6B pre-trained model was used.
## Training
To train the model(s) in the paper iteratively run through the attached notebook, Final_VAE

The cell that trains is:
```
nb_epoch=70
n_steps = 30000/parameters.batch_size 
for counter in range(nb_epoch):
    print('-------epoch: ',counter,'--------')
    vae.fit(sent_generator(TRAIN_DATA_FILE, parameters.batch_size),steps_per_epoch=n_steps, epochs=1, callbacks=[checkpointer, early_stopping],validation_data=(dat
```
## Evaluation
Please read report for evaluation considerations.

## Results
The model produced one line of text repeated

## Files in this Repository
We included some non-final implementations to highlight other approaches taken. These are VAE_Text_Generation (Colab baseline).py, training_function.py, vae.py, vae_run.py. All these files include iterative trial and errors. 

## Overleaf Link - Final
https://www.overleaf.com/read/gssrgmyfdkqm


