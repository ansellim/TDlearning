# CS7642 Project 1 Readme

## Authorship information

Author: Ansel Lim
Date: 20/21 February 2022

## About this project

This project aimed to reproduce Sutton's seminal paper on temporal difference learning (TD learning).
In Sutton's paper, he introduced a simple dynamical system with bounded random walks which end either in a terminal state with reward 1 or a 
terminal state with reward 0. No rewards are obtained along the way, and the agent does not have any choice in the actions that it can take at any 
stage; with the exception of states adjacent to terminal states (in which case the agent will move to the corresponding terminal state), the agent 
must randomly walk from one state to its neighboring states with equal probability. We will implement TD-lambda learning on this simple system and 
demonstrate how TD(lambda) learning fares for different values of lambda (discount factor) and alpha (learning rate), in comparison with the true 
predictions of ending up in the terminal state with reward 1.

## About the code

The code for experiments 1 and 2 are in the file `experiments.py`. 
The code was extracted from the Jupyter notebook `experiments.ipynb` which contains the code for both experiments.
You may read either the code in the .py file or the Jupyter notebook.
The comments in the code contain detailed information regarding the methodology.
The code should be read in conjunction with the project report.

The code takes a pretty long time to run. However, the functions for optimization/learning the weights allow for easy prototyping if you limit the 
range of alphas and lambdas (for experiments 1 and 2), and limit the maximum possible number of iterations before a lambda-alpha configuration is 
deemed to have failed to converge (for experiment 1).

## Requirements for running the code

I recommend the following environment and dependencies:

* python, version 3.7 or later
* numpy 1.21.1
* matplotlib 3.4.3
