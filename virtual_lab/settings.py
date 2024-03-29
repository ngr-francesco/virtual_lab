from virtual_lab.logger import LogLevel
from matplotlib.cm import Accent,tab10
from matplotlib.colors import TABLEAU_COLORS
from itertools import chain
class Preferences:
    point_selection = ["last","last_before_event","before_event","after_event"]
    plot_directory = "plots/"
    user_prefs_directory = "usr_prefs/"
    default_constant_directory = "const"
    _constants_directory = "const"
    default_log_level = LogLevel.CRITICAL
    color_presets = {"var": list(chain(Accent.colors,tab10.colors)), "event": list(TABLEAU_COLORS.values())}
    exp_T = 1000
        

    def set_constants_directory(self, path):
        self._constants_directory = path
    
    # TODO: Allow automatic detection of custom constants file to select default constants for class initialization.
    constants_directory = property(fget = lambda self: self._constants_directory, fset = set_constants_directory) 
    default_figsize = (6,6.5 ) #TODO implement this in the simulation class!

    def determine_figsize(self,nrows:int ,ncols:int,use_legend: bool = False, separate_legend = False):
        legend_space = 2 if (use_legend and not separate_legend) else 0
        figsize = (self.default_figsize[0]*ncols+legend_space,self.default_figsize[1]*nrows)
        return figsize

prefs = Preferences()