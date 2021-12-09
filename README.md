# Generate Prune Select - CISC 867 TensorFlow Reproduction

## Requirements
To install requirements:
```setup
pip install tensorflow version==2.6.0
pip install tensorflow-addons
pip install numpy
pip install scipy
pip install -r orig_requirements.txt
```
## Note
utils.py, Main.py, languagequality.py, and the utility folder were not modified by us - these are from the original repo.
Similairly Data/... and utility/.. are run requirements from the original.

The GloVe embeddings can be downloaded from https://nlp.stanford.edu/projects/glove/, the 6B pre-trained model was used.
## Training
To train the model(s) in the paper iteratively run through the attached notebook, Final_VAE

The cell that trains is:
```
nb_epoch=70
n_steps = 30000/parameters.batch_size 
for counter in range(nb_epoch):
    print('-------epoch: ',counter,'--------')
    vae.fit...
## Evaluation
Please read report for evaluation considerations.


## Results
The model produced one line of text repeated

## Files in this repo
We included some non-final implementations to highlight other approaches taken. These are VAE_Text_Generation (Colab baseline).py, training_function.py, vae.py, vae_run.py. All these files include iterative trial and error efforts towards the final submission. 
The saved model (.h5 format) can be accessed here: https://drive.google.com/file/d/1Zbtag_JezIdmdtwNq3rFBAWW6qUZQkgK/view?usp=sharing

## Overleaf Link - Final
https://www.overleaf.com/read/gssrgmyfdkqm
