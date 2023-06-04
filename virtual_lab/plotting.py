from .simulation import Simulation
import matplotlib.pyplot as plt
from .settings import prefs

class Figure:
    def __init__(self,
                 figsize = prefs.default_figsize, 
                 name = "fig",
                 n_rows = 1,
                 n_cols = 1,
                 **kwargs):
        """
        A class meant to manage figures and plots. 
        It should be easy to add new figures and redo the plot also from jupyter notebooks.
        The figure class is kept "alive" during the simulation, so that it can be updated and modified
        without the need to create a new figure every time.
        """
        self.figsize = figsize
        self.name = name
        self.n_rows = n_rows
        self.n_cols = n_cols
    
    def plot(self):
        """
        This method collects all of the variables and data that are needed to plot the figure.
        It is called every time the figure is updated.
        """
        pass

    def add_title(self, title):
        self.title = title
    
    def add_xlabel(self, xlabel):
        self.xlabel = xlabel
    
    def add_ylabel(self, ylabel):
        self.ylabel = ylabel
    
    def add_legend(self, legend):
        self.legend = legend
    