<img src="logo/1@2x.png" alt="Virtual Lab logo" width="300"/>
# VIRTUAL LAB

Virtual Lab is a simulation manager in which the user can define 3 main sets of objects:
- `Experiment` objects which contain the details of the experiments you want to run
- `Model` objects which contain the specific behavior of the system you are modelling (e.g. differential equations, stochastic processes, deterministic equations)
- `Simulation` objects, which can manage `Experiment`s and `Model`s and run the experiments on the selected models

The idea behind virtual lab is to make it easy to try various iterations of a possible theoretical model without the need to change the whole experimental script, and viceversa to be able to add any number of experiments behind the scenes and simply run your simulation by pointing to the specific experiment you want to try. 
This, hopefully results in more clean and organized scripts which will save you a precious time during the endless coding sessions for your computational models.

Examples of how to use these classes are shown in the virtual_lab.examples folder as jupyter notebooks.

Dependencies:
- python 3.7
- numpy
- matplotlib
