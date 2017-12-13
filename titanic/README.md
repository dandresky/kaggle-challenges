## Competition Description

From the Kaggle site:

"The sinking of the RMS Titanic is one of the most infamous shipwrecks in history.  On April 15, 1912, during her maiden voyage, the Titanic sank after colliding with an iceberg, killing 1502 out of 2224 passengers and crew. This sensational tragedy shocked the international community and led to better safety regulations for ships.

"One of the reasons that the shipwreck led to such loss of life was that there were not enough lifeboats for the passengers and crew. Although there was some element of luck involved in surviving the sinking, some groups of people were more likely to survive than others, such as women, children, and the upper-class.

"In this challenge, we ask you to complete the analysis of what sorts of people were likely to survive. In particular, we ask you to apply the tools of machine learning to predict which passengers survived the tragedy."

## Source Code and Data

- [titanic.ipynb](titanic.ipynb) is a Jupyter notebook with all of my notes and analysis regarding the data set.
- [src/titanic.py](src/titanic.py) is the main Python script file that launches all data processing and modeling. It includes scripts to read in both the training and test data sets, make modifications identified in the Jupyter notebook, and start the modeling and cross-validation process.    
- [src/model_parameters.py](src/model_parameters.py) contains functions that construct parameter permutation directories for each type of model evaluated. These dictionaries are fed to the sklearn GridSearchCV object which is used to find the optimal parameters for the model under test.
- [src/cross_val.py](serc/cross_val.py) contains functions to implement GridSearchCV and other cross validation methods as well as functions to score and report on results. 
