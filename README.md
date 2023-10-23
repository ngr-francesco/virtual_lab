<img src="logo/1@2x.png" alt="Virtual Lab logo" width="300"/>

# VIRTUAL LAB

Virtual Lab is a simulation manager software which has the purpose of simplifying the pipeline involved in testing scientific computational models. The main idea behind virtual lab is to organize the workflow into separate steps in order to keep the user scripts as clean as possible:
1. First, defining our `Model`, which can contain any number of time dependent functions, differential equations or any of the currently supported stochastic processes. The default values of the parameters used in the model can be given at initialization and will then be available as class attributes.
2. Then defining some `Experiment`s, which describe the experimental conditions under which we want to analyse our model during our simulations. Each experiment defines a set of events which will correspond to time dependent modifications to the parameters of our model.
3. Finally, we initialize our `Simulation`, which can apply the `Experiment`'s conditions to our `Model`, run simulations and analyse the results. Our `Simulation` can contain any number of `Model`s and can readily compare the results obtained on the same `Experiment` from different models.

This procedure cleanly organizes the workflow into different tasks, each related to a relevant moment in the development of a computational model. With the `Model` class, we can define any number of different variations of the computational model we intend to test without the need to overwrite or copy-paste our code, by simply subclassing the main version and introducing only the specifically required modifications. These models can then simply be added to the list of models which are being managed by the `Simulation` class, and switching from one to the other is a matter of a single line of code (see the toy [example](https://github.com/ngr-francesco/virtual_lab/blob/main/examples/toy_example.ipynb)). The `Simulation` class is equipped with various useful tools both for the computation of our results as well as for managing the results and displaying them into insightful plots. For example, if a model has a stochastic component, it is possible to choose if some time should be given to the model to equilibrate with the stochastic process before running the experiment, or if this relaxation process should be included in the results. 

The `Experiment` class is particularly usefull when our model should be tested under many different experimental conditions. This class can be used to define a library of different experiments which can then be grouped under the larger `Experiments` class. Then these `Experiments` can be tested on all of the variations of our model by using, again, a single line of code. Experiments can be easily managed through these classes so that subsets can be tested or modified without requiring to overwrite code, and if one experiment differs only slightly from another, it is easy to duplicate it and change the required parameters without cluttering the code by copy pasting.

The plotting capabilities of the `Simulation` class include:

- Variable specific color coding: if various versions of our model use the same variable, this variable will always appear with the same color coding. This color coding is saved at the end of each session, meaning that it will be permanently set. If the color coding should be changed this can be done through the apposite method or through the manual modification of the user preferences file.
- Simple plotting of the results obtained from any number of experiments. One can easily analyse the results of multiple simulations by plotting a single figure in a single line. 
- Multiple models can be compared within the same figure over any number of experimental results. Each model occupying a separate column in order to make the comparison easier. It is also possible to display the equations defining the model in the corresponding column, if these equations have been previously specified by the user.
- If we wish to only plot a selection of points from an experiment corresponding to a specific event from the `Experiment` class, we can simply specify the type of event and the `Simulation` will handle the rest. 
- The same can be applied if we want to observe the effect of the same event over different experiments, and we can plot the value of some of the model's variables as a function of the experiment.


The idea behind virtual lab is to make it easy to try various iterations of a possible theoretical model without the need to change the whole experimental script, and viceversa to be able to add any number of experiments behind the scenes and simply run your simulation by pointing to the specific experiment you want to try. 
This, hopefully results in more clean and organized scripts which will save you a precious time during the endless coding sessions for your computational models.

Examples of how to use these classes are shown in the virtual_lab.examples folder as jupyter notebooks.

Dependencies:
- python 3.7
- numpy
- matplotlib
