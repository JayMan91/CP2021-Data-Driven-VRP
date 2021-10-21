# CP2021-Data-Driven-VRP

## Overview
This repository contains the code and the data of the paper:
**Data Driven VRP: A Neural Network Model to learn hidden preferences for VRP**. *27th International Conference on Principles and Practice of Constraint Programming (CP 2021)*. 

**Bibtex formatted citation**
```
@InProceedings{mandi_et_al:LIPIcs.CP.2021.42,
  author =	{Mandi, Jayanta and Canoy, Rocsildes and Bucarey, V{\'\i}ctor and Guns, Tias},
  title =	{{Data Driven VRP: A Neural Network Model to Learn Hidden Preferences for VRP}},
  booktitle =	{27th International Conference on Principles and Practice of Constraint Programming (CP 2021)},
  pages =	{42:1--42:17},
  series =	{Leibniz International Proceedings in Informatics (LIPIcs)},
  ISBN =	{978-3-95977-211-2},
  ISSN =	{1868-8969},
  year =	{2021},
  volume =	{210},
  editor =	{Michel, Laurent D.},
  publisher =	{Schloss Dagstuhl -- Leibniz-Zentrum f{\"u}r Informatik},
  address =	{Dagstuhl, Germany},
  URL =		{https://drops.dagstuhl.de/opus/volltexte/2021/15333},
  URN =		{urn:nbn:de:0030-drops-153339},
  doi =		{10.4230/LIPIcs.CP.2021.42},
  annote =	{Keywords: Vehicle routing, Neural network, Preference learning}
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


