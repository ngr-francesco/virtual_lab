import pytest 
from virtual_lab import Simulation, Model, prefs

def test_color_coding():
    """
    Initializes a Simulation and a Model with a number of variables which exceeds the 
    number of colors defined in settings.prefs. The color_coding of the Simulation should then be set into
    "cycle" and "monocrome" modes and the colors should either be cycled through or all set to black.
    So to check if the test is successful, we check if the colors are either all black or if they are cycling through the preset colors
    in the ColorCoding class.
    """
    # Create a Simulation and a Model
    sim = Simulation()
    N_colors = len(prefs.color_presets["var"])+10
    model = Model("model", {f"var_{k}": 0 for k in range(N_colors)})

    # Create a number of variables which exceeds the number of colors defined in settings.prefs
    for i in range(len(model.variables)):
        model.add_variable(f"var{i}", 0)
    # Add the model to the simulation
    sim.add_model(model)
    # Check if the color_coding is set to "cycle" mode
    assert sim.color_coding.mode == "cycle"
    # Check if the colors are cycling through the preset colors in the ColorCoding class
    assert sim.color_coding.colors == prefs.colors
    # Set the color_coding to "monochrome" mode
    sim.color_coding.mode = "monochrome"
    # Check if the color_coding is set to "monochrome" mode
    assert sim.color_coding.mode == "monochrome"
    # Check if the colors are all set to black
    assert sim.color_coding.colors == ["#000000"]

