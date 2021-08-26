# CP2021-Data-Driven-VRP

## Overview
This repository contains code for the paper:
Data Driven VRP: A Neural Network Model to learn hidden preferences for VRP. 27th International Conference on Principles and Practice of Constraint Programming (CP 2021). 

**Bibtex formatted citation**
```
@article{mandi2021data,
  title={Data Driven VRP: A Neural Network Model to Learn Hidden Preferences for VRP},
  author={Mandi, Jayanta and Canoy, Rocsildes and Bucarey, V{\'\i}ctor and Guns, Tias},
  journal={arXiv preprint arXiv:2108.04578},
  year={2021}
}
```
## Running the Experiment

To run the experiment first first unzip `data.zip`.
Install the required packages
```
pip install -r requirements.txt
```

Run the Markov models by running the command
```
python MarkovExp.py
```
Run the Neural network models by running the command 
```
python NeuralNetExp.py
```
Run the decision focused model by running the command
```
python DecisionFocusedExp.py
```


