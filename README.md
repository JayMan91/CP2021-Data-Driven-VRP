# CP2021-Data-Driven-VRP

## Overview
This repository contains the code and the data of the paper:
**Data Driven VRP: A Neural Network Model to learn hidden preferences for VRP**. *27th International Conference on Principles and Practice of Constraint Programming (CP 2021)*. 

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

To run the experiment first unzip `data.zip` by running
```
unzip data.zip
```

Then install the required packages
```
pip install -r requirements.txt
```
Then to reproduce the experimental results, 

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


