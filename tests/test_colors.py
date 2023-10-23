import pytest 
from virtual_lab import Simulation, Model, prefs
from virtual_lab.const import ColorCodingStrategies

def test_color_coding():
    """
    Initializes a Simulation and a Model with a number of variables which exceeds the 
    number of colors defined in settings.prefs. The color_coding of the Simulation should then be set into
    "cycle" and "monocrome" modes and the colors should either be cycled through or all set to black.
    So to check if the test is successful, we check if the colors are either all black or if they are cycling through the preset colors
    in the ColorCoding class.
    """
    # Create a Simulation and a Model with the default color coding
    sim = Simulation(load_preferences = False)

    # Default case (not reliant on the model)
    assert sim.color_coding.strategy == ColorCodingStrategies.CYCLE
    # Check if the colors were loaded correctly
    assert sim.color_coding.available_colors == prefs.color_presets

    N_colors = len(prefs.color_presets["var"])
    model = Model("model", {f"var_{k}": 0 for k in range(N_colors + 1)})
    # Add the model to the simulation
    sim.add_model(model)

    # Check after adding a new model with the correct number of variables to induce one cycle:
    assert sim.color_coding.available_colors['var'] == prefs.color_presets['var']

    # Create a number of variables which exceeds the number of colors defined in settings.prefs
    for i in range(len(model.variables) - 1):
        model.add_variable(f"var{i}", 0)

    # Check if the colors are cycling correctly
    assert sim.color_coding.available_colors['var'] == prefs.color_presets['var']

    sim.color_coding.monocrome()

    assert sim.color_coding.strategy == ColorCodingStrategies.MONOCROME
    # Check if the colors are all set to black
    for var_type in sim.color_coding.available_colors:
        assert sim.color_coding.available_colors[var_type] == ["#000000"]


if __name__ == '__main__':
    test_color_coding()

