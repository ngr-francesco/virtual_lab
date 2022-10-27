# virtual_lab

This is a small software package I developed while working on my Thesis. It's a simulation manager in which the user can define 3 main sets of objects:
- `Experiment` objects which contain the details of the experiments you want to run
- `Model` objects which contain the specific behavior of the system you are modelling (e.g. differential equations, stochastic processes, deterministic equations)
- Simulation objects, which can manage `Experiment`s and `Model`s and run the experiments on the selected models

The idea behind this is that it should easy to try various iterations of a possible theoretical model without the need to change the whole experimental script, and viceversa to be able to add any number of experiments behind the scenes and simply run your simulation by pointing to the specific experiment you want to try. 
This, hopefully results in more clean and organized scripts which will save you a bunch of time during your endless coding sessions for your computational models.
If I will feel compelled to continue adding functionalities I will also start making a proper documentation and some basic tutorials. For now you can see how I've been using _virtual lab_ by downloading the jupyter notebook in the examples folder.
