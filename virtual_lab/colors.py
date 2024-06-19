from .logger import get_logger
from .settings import prefs
from copy import deepcopy
from virtual_lab.const import ColorCodingStrategies
import json

# TODO: the current color scheme is not applicable in the case in which you want to compare models
# on the same plot (they would have the same color coding). 
# Not a big problem since I think we never actually plot different models in the same plot (should be done outside of the Simulation class by the user)
# But could be a nice additional feature, to just say Sim.compare_model_variables([var1,var2]) and get a nice plot.

class ColorCoding:
    """
    Class to handle color coding of variables.
    It displays the color coding in a table and allows the user to change the color coding.
    """
    def __init__(self, color_coding: dict = {}, color_presets: dict = prefs.color_presets):
        self.color_coding = color_coding
        self._color_presets = color_presets
        self.available_colors = {key: [] for key in self.color_presets.keys()} 
        # Set the default strategy for running out of colors       
        self.cycle()
        self.cycling_index = 0
        self.logger = get_logger("ColorCoding")
        self.logger.debug(f"Initialized color coding with {color_coding}")
    
    color_presets = property(fget = lambda self: self._color_presets, fset = lambda self, value: self.logger.warning("Cannot set color presets."))

    def cycle(self):
        """
        Set the strategy used when running out of available colors to "cycle".
        This strategy will cycle through the preset colors when running out of available colors.
        """
        self.strategy = ColorCodingStrategies.CYCLE
        self.refresh_available_colors()
    
    def monocrome(self):
        """
        Set the strategy used when running out of available colors to "monocrome".
        This strategy will use the same color for all variables when running out of available colors.
        """
        self.strategy = ColorCodingStrategies.MONOCROME
        self.refresh_available_colors()


    def __repr__(self):
        """
        Shows the color coding in a table.
        """
        return self.get_table()

    def get_table(self):
        """
        Returns a table with the color coding.
        """
        table = "Color coding:\n"
        for key, value in self.color_coding.items():
            table += f"{key}: {value}\n"
        return table
    
    def set_var_color(self, key, color):
        """
        Set the color for a variable.
        """
        self._set_color(key, color, "var")
    
    def set_event_color(self, key, color):
        """
        Set the color for a variable.
        """
        self._set_color(key, color, "event")
    
    def _set_color(self,key,color,category = "var"):
        """
        Set the color for a variable.
        """
        # Check if the variable is already in the color coding
        if key in self.color_coding:
            if color == self.color_coding[key]:
                return False
            self.logger.warning(f"Overwriting color for {key}.")
            # If we're changing the color of this var, we add its previous color to the available colors
            self.available_colors[category].append(self.color_coding[key])

        # If the chosen color is in the available colors, we remove it from the available colors
        if color in self.available_colors[category]:
            self.available_colors[category].remove(color)
            self.check_if_run_out_of_colors()

        self.color_coding[key] = color
        return True
    
    def check_if_run_out_of_colors(self):
        """
        Check if we run out of colors.
        """
        for var_type, colors in self.available_colors.items():
            if len(colors) == 0:
                self.manage_empty_available_colors(var_type)
                self.logger.warning(f"Ran out of colors for {var_type}. Consider adding more colors to the available colors.")
    
    def add_default_colors(self, keys, var_type = None):
        """
        Add default colors to the color coding. Default colors are applied only to variables that weren't already being tracked.
        """
        changed_color_coding = []
        if var_type is None:
            var_type = "var"
        for key in keys:
            if not key in self.color_coding:
                self._set_color(key, self.available_colors[var_type][0], var_type)

        
    def refresh_available_colors(self):
        """
        If we run out of colors, we try to refresh the available colors by checking if any colors from the presets are not being used.
        This can happen if the user has changed the color coding manually.
        If we're using the monocrome strategy we simply set the color to black for all variable types.
        """
        if self.strategy == ColorCodingStrategies.CYCLE:
            for var_type, colors in self.color_presets.items():
                for color in colors:
                    if color not in self.color_coding.values():
                        self.available_colors[var_type].append(color)
        elif self.strategy == ColorCodingStrategies.MONOCROME:
            for var_type in self.available_colors:
                self.available_colors[var_type] = ["#000000"]
    
    def manage_empty_available_colors(self, var_type: str = None):
        """
        If we run out of colors for a var_type, we try to refresh the available colors by checking if any colors from the presets are not being used.
        If there are still no colors available, we define the color coding depending on the strategy.
        Parameters:
        ------------
            var_type: str, optional
                The type of variable for which we ran out of colors. If None, we refresh all available colors.
        """
        if var_type is None:
            for var_type in self.available_colors.keys():
                self.manage_empty_available_colors(var_type)
        else:
            self.refresh_available_colors()
            if len(self.available_colors[var_type]) == 0:
                if self.strategy == ColorCodingStrategies.CYCLE:
                    self.available_colors[var_type] = deepcopy(self.color_presets[var_type])
                    self.logger.warning(f"Ran out of colors for {var_type}. Cycling through the available colors.")
                elif self.strategy == ColorCodingStrategies.MONOCROME:
                    self.available_colors[var_type] = ["black"]
                    self.logger.warning(f"Ran out of colors for {var_type}. Using monocrome.")
                else:
                    raise ValueError(f"Unknown strategy {self.strategy}.")
    
    def get_next_color(self, var_type = None):
        """
        Keep track of the cycling index and return the next color from the preset colors.
        """
        if var_type is None:
            var_type = "var"
        if self.cycling_index >= len(self.color_presets[var_type]):
            self.cycling_index = 0
        color = self.color_presets[var_type][self.cycling_index]
        self.cycling_index += 1
        return color
    
    def reset_color_cycle(self):
        """
        Reset the cycling index.
        """
        self.cycling_index = 0
    
    def __getitem__(self, key):
        """
        Get the color of a variable.
        """
        if isinstance(key, list):
            return {k: self.color_coding[k] for k in key}
        
        return self.color_coding[key]
    
    def __contains__(self, key):
        """
        Check if a variable has a color.
        """
        return key in self.color_coding
    
