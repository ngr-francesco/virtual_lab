
class Preferences:
    point_selection = ["last","last_before_event","before_event","after_event"]
    plot_directory = "plots/"
    default_constant_directory = "const"
    _constants_directory = "const"
    def set_constants_directory(self, path):
        self._constants_directory = path
    # TODO: Allow automatic detection of custom constants file to select default constants for class initialization.
    constants_directory = property(fget = lambda self: self._constants_directory, fset = set_constants_directory) 
    default_figsize = (6,5 ) #TODO implement this in the simulation class!
    
    



prefs = Preferences()