import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from warnings import warn
from os import makedirs
from copy import deepcopy
from itertools import chain
from time import perf_counter
import json

from virtual_lab.experiments import Experiments, Experiment
from virtual_lab.settings import prefs
from virtual_lab.model import Model
from virtual_lab.utils import select_unique_handles_and_labels, get_time_unit_in_seconds
from virtual_lab.logger import get_logger, LogLevel
from virtual_lab.colors import ColorCoding
    
class Simulation:
    
    def __init__(self, model = None, **kwargs):
        """
        Simulation class to handle simulation protocols.
        Parameters:
        -----------
            constants: dict, optional
                The constants used during the experiments. Defaults to the 
                constants that were defined in the model
        """

        # Set up the management objects
        self.prefs = kwargs.get("settings",prefs)
        self.logger = get_logger("Simulation")
        self.color_coding = ColorCoding()

        # Set up the simulation-related attributes
        if model is not None:
            self.model = model
            self.models = {model.name: model}
        else:
            self.models = {}
            self.logger.info("Initializing a simulation without a given model. A model should be added before running any experiment"
            " by calling Simulation.add_model()")
        self._ordered_quantities = []
        self.model_results = {}
        load_prefs = kwargs.get("load_preferences",True)
        if load_prefs:
            self.load_user_prefs()    
    
    def save_user_prefs(self):
        makedirs(self.prefs.user_prefs_directory[:-1],exist_ok=True)
        color_coding = self.color_coding.color_coding
        var_colors = self.color_coding.available_colors["var"]
        event_colors = self.color_coding.available_colors["event"]
        dict_to_save = {
            "color_coding": color_coding,
            "var": var_colors,
            "event": event_colors,
            "_ordered_quantities": self._ordered_quantities
        }
        self.logger.debug(f"Saving user preferences to {self.prefs.user_prefs_directory+'user_prefs.json'}")
        with open(self.prefs.user_prefs_directory+"user_prefs.json","w+") as file:
            json.dump(dict_to_save,file, indent = 4)
    
    def load_user_prefs(self):
        """
        Load user preferences from a json file.
        """
        try:
            self.logger.debug(f"Loading user preferences from {self.prefs.user_prefs_directory+'user_prefs.json'}")
            with open(self.prefs.user_prefs_directory+"user_prefs.json","r") as file:
                prefs = json.load(file)
            self._ordered_quantities = prefs["_ordered_quantities"]
            self.color_coding.color_coding = prefs["color_coding"]
            self.color_coding.available_colors["var"] = prefs["var"]
            self.color_coding.available_colors["event"] = prefs["event"]
            self.logger.debug(f"Loaded user preferences: {prefs}")    
        except FileNotFoundError:
            self.logger.debug("No user preferences file found, using default preferences.")

    def add_model(self,model, switching = True):
        """
        Add a model to the simulation.
        """
        if isinstance(model,Model):
            self.models[model.name] = model
            if switching:
                self.logger.info(f"Switching to model {model.name}")
                self.model = model
            # Add a color coding for this model's variables
            self._add_colors(model.variables.varnames)
        else:
            raise TypeError(f"Model must be of type Model, not {type(model)}")
    
    def _add_colors(self,varnames, var_type = "var"):
        """
        Add colors to the color coding for the given variables.
        Parameters:
        -----------
            varnames: list
                The variables for which to add colors
            event: bool, optional   
                If True, the colors will be added to the event colors.
                If False, the colors will be added to the default colors.   
            additional: bool, optional
                If True, the colors will be added to the additional colors.
                If False, the colors will be added to the default colors.  
        """
        self.color_coding.add_default_colors(varnames, var_type=var_type) 
        self.save_user_prefs()
    
    def set_variable_color(self,var,color):
        """
        Set the color for the given variable.
        Parameters:
        -----------
            var: str
                The variable for which to set the color
            color: str
                The color to set
        """
        self.color_coding.set_var_color(var,color)
        self.save_user_prefs()
    
    def get_variable_color(self,var):
        """
        Get the color for the given variable.
        Parameters:
        -----------
            var: str
                The variable for which to get the color
        """
        return self.color_coding[var]

    def switch_model(self,model):
        if isinstance(model,Model):
            self.model = self.models[model.name]
        elif isinstance(model,str):
            self.model = self.models[model]
    
    def termalize_system(self,termalization_time,exp,stochastic = False,record = False):
        """
        I'm not sure this function is needed anymore. Could just put it into the run_experiments() method. I'll consider it.
        Notes:
            Right now stochastic processes are run independently of the actual state of the system (i.e. the quantities they
            depend on are either constants or purely time dependent). For systems in which the stochastic process is not easily 
            retraced to either a simple birth-death process or a random walk, they should be manually implemented in the Model class.
        """
        was_recording = self.model.is_recording
        self.model.is_recording = record
        # Set constants to their basal values (read from experiment/model constants with the '_0' suffix)
        # This assures us that when we run the stochastic processes we'll be using the base values without perturbations
        temp_T = exp.T
        exp.T = termalization_time
        # Prepare the constants to have a sufficient size for running termalization
        self.model.prepare_constants(exp,termalizing = True)
        if stochastic:
            self.model.run_stochastic_processes()
        if hasattr(self.model,"termalization_step"):
            term_func = self.model.termalization_step
        else:
            term_func = self.model.single_step
        
        for t in range(exp.steps):
            term_func(t)
        # Since we only want to termalize once in most cases, copy the current variable values into their init values
        # so that at the next iteration we can start from there.
        self.model.update_variables_init_values()
        # Now we can set the constants back to their values needed for the experiment
        exp.T = temp_T
        # We exclude the stochastic variables because in our constants we might have an initialization value
        # in case the model can also be used as a mean field approximation.
        self.model.prepare_constants(exp,exclude_stochastic = stochastic)
        self.model.is_recording = was_recording

    def run_experiments(self, experiments, n_of_runs=1, 
                        stochastic = False,
                        termalize = False, termalization_time = 2*3600, 
                        record_termalization = False, save_results = True):
        # Format the experiments correctly
        error_msg = """The experiments must be a list of Experiment objects, a single Experiment object,
                     or an instance of the Experiments class, instead, they are of type """
        if isinstance(experiments,list):
            for experiment in experiments:
                if not isinstance(experiment,Experiment):
                    raise TypeError(error_msg + str(type(experiment)))
            experiments = Experiments(experiments)
        elif not (isinstance(experiments,Experiments) 
                or isinstance(experiments,Experiment)):
            raise TypeError(error_msg + str(type(experiments)))
        if isinstance(experiments,Experiment):
            experiments = Experiments(experiments)
        if termalize and not stochastic:
            warn("You chose to termalize your model with a non-stochastic model, "
            "often in deterministic models you could compute your stationary state analytically and save computation time.")
        # Results dictionary
        res_dict = {
            "name" : None,
            "experiment": None}
        variables_to_record = [name for name in self.model.variables.recorded_variables]
        res_dict.update({var : None for var in variables_to_record})
        if self.model.stochastic_variables is not None:
            res_dict.update({var: None for var in self.model.stochastic_variables})
        # Our results are made of a list of dictionaries, one for each experiment.
        results = [res_dict.copy() for k in range(len(experiments))]
        start_time = perf_counter()
        termalized = False
        for i, exp in enumerate(experiments):
            # Make it simple to change nucleation rates for single experiments
            results[i]["name"] = exp.title
            results[i]["experiment"] = exp
            results[i]["model"] = self.model.name
            self.model.dt = exp.dt # TODO: Better management of the dt

            self.model.prepare_constants(exp,exclude_stochastic = stochastic)
            
            # In case we're running a termalization process we only want to run this once
            # because then re-initialization is automatically taken care of inside of model.reset()
            if not stochastic or not termalize or not termalized:
                self.model.initialise_variables(exp,stochastic)
            for k in range(n_of_runs):
                # INITIALIZE ALL OF THE QUANTITIES FOR THE SIMULATION
                # If we want to allow the system to termalize on its own
                if termalize:
                    self.termalize_system(termalization_time,exp,stochastic, record_termalization)
                if stochastic:
                    self.model.run_stochastic_processes()
                
                # Simulation Loop
                # TODO check if this is considerably faster if done within the model
                for t in range(exp.steps):
                    self.model.single_step(t)
                
                try:
                    for var in variables_to_record:
                        data = self.model.variables.recorded_values[var]
                        results[i][var] = np.row_stack([results[i][var],np.array(data)])
                    # Stochastic variables are kind of a special case so we treat them separately
                    if self.model.stochastic_variables is not None:
                        for var in self.model.stochastic_variables:
                            data = getattr(self.model,var)
                            results[i][var] = np.row_stack([results[i][var],np.array(data)])
                except ValueError:
                    for var in variables_to_record:
                        data = self.model.variables.recorded_values[var]
                        results[i][var] = np.array(data)
                    if self.model.stochastic_variables is not None:
                        for var in self.model.stochastic_variables:
                            data = getattr(self.model,var)
                            results[i][var] = np.array(data)
                # Reset the model variables to their initial state
                self.model.reset()
            if n_of_runs > 1:
                for var in variables_to_record:
                    results[i]["mean_"+var] = np.mean(results[i][var],axis = 0)
                    results[i]["std_"+var] = np.std(results[i][var],axis = 0)
        if save_results:
            self.model_results[self.model.name] = results
        end_time = perf_counter()
        duration = end_time-start_time
        self.logger.info(f"Simulating {len(experiments)} experiments took: {duration:.3f} s")
        plt.show()
        return results

    def run_variable_perturbation(self,varnames,delta):
        """
        Initialize the system in its stationary state (if there is one). Then perturb one of the variables to 
        measure the relaxation process. Particularly useful for models which cannot be integrated analytically.
        """
        raise NotImplementedError
        
    
    def reduce_data(self,values, start,stop,step=1):
        new_values = {}
        for name,data in values.items():
            if isinstance(data,str) or isinstance(data,Experiment):
                continue
            if len(data.shape) > 1:
                for i in range(len(data)):
                    tmp = data[i][slice(start,stop,step)]
                    new_data = np.row_stack([new_data,tmp]) if i > 0 else tmp
                new_values[name] = new_data
            else:
                new_values[name] = data[slice(start,stop,step)]
        return new_values
    
    def model_equations(self,models = None, figure = None, rows = 1,**kwargs):
        if models == None:
            models = [self.model]
        elif isinstance(models,list) and isinstance(models[0],str):
            models = [self.models[k] for k in models]
        elif isinstance(models,str):
            models = [self.models[models]]
        # Only look at the width, the height will be fixed
        figsize = (kwargs.get('figsize',self.prefs.determine_figsize(1,1,False)))
        if figure == None:
            figure = plt.figure(figsize=figsize)
        for i,model in enumerate(models):
            ax = figure.add_subplot(rows,len(models),i+1)
            equations = model.latex_equations()
            equations.insert(0,f"Equations for model: {model.name}")
            cols = len(equations)
            pos = np.arange(0.8,0.1,-0.7/cols)
            for eq,y in zip(equations,pos):
                ax.text(0.5,y,eq,ha = "center",va="center",fontsize = 16)
            ax.set_axis_off()


    def plot_results(self, results = None, exp_names = None, 
                    time_interval = None, filename = "img.pdf", 
                    time_unit = "min", step = 1, show_equations = False, **kwargs):
        """
            Plot the volumes computed in the experiments.
        Parameters:
        ----------
            results (list(dict)): containing the results of the experiments in a dictionary,
                        generated by the method simulation_routine()
            exp_names (list(int)): the list of the experiments to be plotted
            time_interval (tuple,optional): define the time interval to be plotted, defaults to None
            name (string,optional): the name to give to the output image, defaults to 'img.pdf'
            time_unit (str,optional): whether the time axis should be given in minutes, seconds or hours, defaults to True
            step (int,optional): step used when slicing the data for the plot, defaults to 1 (all the data)
        
        """
        if results is None:
            results = self.model_results[self.model.name]
        if time_interval is not None:
            assert len(time_interval) == 2
        from math import ceil
        if exp_names is not None:
            counter = 0
            for exp in exp_names:
                for result in results:
                    if exp == result["experiment"].name:
                        counter += 1
        n_plots = len(results) if exp_names is None else counter
        self.logger.info(f"Plotting {n_plots} experiments")
        use_legend = kwargs.get('legend', True)
        separate_legend = kwargs.get('separate_legend', False)
        plot_cols = kwargs.get("n_cols",min(n_plots,4))
        plot_rows = ceil(n_plots/plot_cols)
        figsize = kwargs.get('figsize',self.prefs.determine_figsize(plot_rows,plot_cols,use_legend,separate_legend))
        fontsize = kwargs.get('fontsize',12)
        use_title = kwargs.get('use_title',True)
        variables = kwargs.get('variables',None)
        lw = kwargs.get('lw',3)
        fig = plt.figure(figsize=figsize)
        plt.rcParams['font.size'] = fontsize
        # Taking care that the experiments to be plotted are in the right format
        if exp_names is None:
            exp_names = [result["name"] for result in results]
        else:
            if isinstance(exp_names,list):
                exp_names = [str(experiment) for experiment in exp_names]
            elif isinstance(exp_names, Experiments):
                exp_names = str(exp_names)
            elif isinstance(exp_names,str) or isinstance(exp_names,Experiment):
                exp_names = [str(exp_names)]
        # Give a warning if the experiment name is not in the results
        result_names = [result["name"] for result in results]
        for experiment in exp_names:
            if not experiment in result_names:
                warn("The experiment you're trying to plot is not among the results! Maybe you forgot to run the simulation?")
        if time_interval is not None:
            try:
                if len(time_interval[0]) > 0:
                    assert len(time_interval) == len(exp_names)
            except TypeError:
                tmp = [time_interval]
                for k in range(len(exp_names)-1):
                    tmp.append(time_interval)
                time_interval = tmp
        plot_handles,plot_labels = np.array([]),np.array([])
        plot_idx = 0
        add_col_idx = 0
        rows = ceil(len(results)/plot_cols)
        cols = (plot_cols if len(results) >= plot_cols else len(results))
        rows = rows+1 if show_equations else rows
        start_index = 1 + cols if show_equations else 1
        if show_equations:
            models = []
            for result in results:
                if not result["model"] in models:
                    models.append(result["model"])
            self.model_equations(models,fig,rows)
        axes = []
        for result in results:
            if result["name"] in exp_names:
                # TODO generalize this so it can be used in any plot by using kwargs.
                model_name = result["model"]
                exp = result["experiment"]
                idx = exp_names.index(result["name"])
                interval = time_interval[idx] if time_interval is not None else None
                time_axis,timescale,start,stop,_ = self._get_time_quantities(exp,interval,step,time_unit)
                values = self.reduce_data(result,start,stop,step)
                if variables is None:
                    model_variables = self.models[model_name].variables
                    model_variables = [var for var in model_variables if var in model_variables.recorded_variables]
                else:
                    model_variables = variables
                min_value = self.get_min_value(values,variables = model_variables,skip_means = False)
                # If we have saved a 'mean' then also an 'std' for sure will be there.
                plot_std = sum(['mean_' + name in values.keys() for name in model_variables]) == len(model_variables)

                # Plotting starts here
                reference_axis = int(plot_idx/cols)*cols
                axes.append(fig.add_subplot(rows,cols,plot_idx+start_index,sharey=axes[reference_axis] if plot_idx%cols!=0 else None))
                if use_title:
                    plt.title(model_name + ": " + result["name"])
                # FIXME: This is not generally useful, remove!
                plt.plot([start,stop/timescale],[1,1] ,color='0', ls=":")

                for name in model_variables:
                    if name in values.keys():
                        ls = '-'
                        if "ref" in name:
                            ls = ':'
                        try:
                            col = self.color_coding[name]
                        except KeyError:
                            self._add_colors([name])
                            col = self.color_coding[name]
                        var_label = self.determine_label(name)
                        if plot_std:
                            plt.plot(time_axis, values["mean_" + name], label = var_label, lw = lw,color = col,ls = ls)
                            plt.fill_between(time_axis,values["mean_" + name] - values["std_"+name],
                                                    values["mean_" + name]+ values["std_" + name],alpha = 0.3,color = col)
                        else:
                            plt.plot(time_axis,values[name],label = var_label,lw = lw, color = col,ls = ls)
                if exp.experimental_data is not None:
                    self.plot_experimental_data(exp,values,plot_std,timescale)

                # Plotting the quantities used in our experiment
                self.plot_experiment_quantities(exp,min_value,model_name,time_axis[-1],timescale)
                
                # Styiling and legend
                # Don't put the ticks on the part of the quantities since they don't mean anything
                yticks = plt.yticks()[0]
                yticks = [k for k in yticks if k >= min_value]
                # TODO: remove this line!
                yticks.insert(0,0)
                plt.yticks(yticks)
                plt.xlabel(f"t in {time_unit}")
                plt.ylabel(kwargs.get("ylabel",r"Volume in $\mu m^3$"))
                # # REMOVE THIS!
                # if result["model"] == "PSD Layering":
                #     plt.ylim(top = 1.6)
                #     plt.yticks([0.0,0.5,1.0,1.5])
                if use_legend:
                    handles,labels = axes[plot_idx].get_legend_handles_labels()
                    plot_handles= np.append(plot_handles,handles)
                    plot_labels= np.append(plot_labels, labels)
                plot_idx += 1

        # Save figure in the plots directory
        if use_legend:
            plot_handles,plot_labels = select_unique_handles_and_labels(plot_handles,plot_labels)
            if not kwargs.get("separate_legend",False):
                plt.legend(plot_handles,plot_labels, bbox_to_anchor = (1,0), loc= 'lower left', ncol=kwargs.get('legend_cols', ceil(len(model_variables)/10)))
                plt.tight_layout()
            else:
                self.plot_separate_legend(plot_handles,plot_labels,filename= filename) 
        makedirs(self.prefs.plot_directory[:-1],exist_ok=True)
        fig.tight_layout()
        fig.savefig(self.prefs.plot_directory + filename)
        self.logger.debug(f"Saved plot figure in file: {filename}")
        plt.show()


    def plot_separate_legend(self,plot_handles = None,plot_labels= None, axis = None, filename = ".png"):
        if plot_handles is None or plot_labels is None:
            plot_handles,plot_labels = axis.get_legend_handles_labels()
        # We want quantities to be always after variables in the legend
        tmp_l,tmp_h = plot_labels.copy(),plot_handles.copy()
        for label in plot_labels:
            if not self.determine_varname(label) in self._ordered_quantities:
                continue
            idx = tmp_l.index(label)
            l = tmp_l.pop(idx)
            h = tmp_h.pop(idx)
            tmp_l.append(l)
            tmp_h.append(h)
        plot_handles,plot_labels = tmp_h,tmp_l
        figl, axl = plt.subplots()
        axl.axis(False)
        ncols = min(ceil(len(plot_labels)/4),2)
        axl.legend(plot_handles,plot_labels, loc="center", bbox_to_anchor=(0.5, 0.5),frameon = True,
                    edgecolor = "k", ncol = ncols)
        figl.tight_layout(pad = 0)
        makedirs(self.prefs.plot_directory[:-1],exist_ok=True)
        figl.savefig(self.prefs.plot_directory + "legend_" + filename, transparent= True,
                    bbox_inches='tight',pad_inches = 0, dpi = 200)
    
    def plot_experimental_data(self,exp,values,plot_std,timescale):
        data_dict = exp.experimental_data
        try:
            renorm_value = values["mean_" + data_dict["renorm_var"]][0] if plot_std else values[data_dict["renorm_var"]][0]
        except KeyError:
            renorm_value = 1
        # TODO: This is a bit terrible
        plot_dict = data_dict.copy()
        plot_dict.pop("renorm_var")
        plot_dict.pop("time_unit")
        self._add_colors(plot_dict.keys())
        for data in plot_dict:
            col = self.color_coding[data]
            # TODO: This is also highly incorrect and should happen in other places
            y_data = np.array(data_dict[data]["y"])*renorm_value
            data_time_to_seconds = get_time_unit_in_seconds(data_dict["time_unit"])
            x_data = np.array(data_dict[data]["x"])*data_time_to_seconds/timescale
            plt.plot(x_data,y_data,label = data,lw = 2,marker = 'x',markersize = 10,ls='--',color = col)
    
    def plot_experiment_quantities(self,exp,min_value,model_name,T,timescale):
        quantities = exp()
        print("exp_quantities", quantities)
        sorted_quantities = self.sort_by_appearance(quantities.keys())
        # This counter is useful for knowing which of the experiments should determine the legend (the one with the most quantities being plotted)
        # and also to know in which positions to plot the quantities so they don't overlap vertically
        # TODO: this part maybe belongs to a separate function
        yticks = plt.yticks()[0]
        value_range = max(yticks)-min_value
        # We want this part of the plot to be in proportion to the rest
        positions_range = value_range/5
        offset = value_range/10
        positions = np.arange(min_value-positions_range-offset,min_value-offset,positions_range/len(sorted_quantities))
        self._add_colors(list(quantities.keys()),var_type = "event")
        # Make a list of the quantities needed for the model, we only want to plot those.
        dep = list(self.models[model_name].quantity_dependencies().values())
        # Flatten the list of the keys contained in the values of the dictionary returned by quantity_dependencies()
        used_quantities = list(chain(*[list(dep[i].keys()) for i in range(len(dep))]))
        self.logger.debug(f"Plotting experimental quantities: {used_quantities}\n{sorted_quantities}")
        for q,pos in zip(sorted_quantities,positions):
            col = self.color_coding[q]
            print("quantities", q)
            if q in used_quantities:
                plt.plot([0,T],[pos,pos], color = col, ls = ":", alpha = 1)
                if quantities[q] is not None and len(quantities[q]):
                    for ton,toff in quantities[q]:
                        toff = min(toff,T*timescale)
                        plt.plot([ton/timescale,toff/timescale], [pos,pos], color = col,lw = 6)
                    q_label = self.determine_label(q)
                    plt.plot([ton/timescale,toff/timescale], [pos,pos],label = q_label,color = col, lw = 6)
    
    def sort_by_appearance(self,names):
        """
        Keep track of which quantities have been encountered before, so that you can keep the ordering consistent
        throughout the simulation.
        """
        for name in names:
            if not name in self._ordered_quantities:
                self._ordered_quantities.append(name)
        return [name for name in self._ordered_quantities if name in names]
    
    def _get_time_quantities(self, exp, time_interval = None, step = 1,time_unit = "min",**kwargs):
        start = time_interval[0] if time_interval is not None else 0
        dt = step*exp.dt
        stop = time_interval[1] if time_interval is not None else int(exp.T) 
        timescale = get_time_unit_in_seconds(time_unit)
        time_axis = np.arange(start,stop,dt)/timescale
        return time_axis,timescale,start,stop,dt
    
    def get_min_value(self,values,variables = None,skip_means = True):
        if variables is None:
            variables = list(values.keys())
        vals = np.array([])
        for var in variables:
            if (not var in values) or "mean" in var:
                continue
            if "std" in var and not skip_means:
                vals = np.append(vals,values["mean_"+var[4:]]-values[var])
            else:
                vals = np.append(vals,values[var])
        return np.min(vals)
        
    
    def compare_experiment_results(self,results,variables, rescale = False, exp_data = False, filename = "img.png", multi_plot = 1, **kwargs):
        """
        Takes different results obtained on the same experiment and compares them in a single plot.
        Important Note: the rescaling option takes the first result in the list as the reference. If the following results contain multiple 
        stochastic simulations, each of these simulations will be rescaled, so their relative distribution will be affected!
        So if you're planning on comparing stochastic and deterministic experiments, and want to keep the true distribution of the stochastic
        outcomes, then put the corresponding experiment fist in line. 
        A more sensible rescaling option will be implemented in the future..

        """
        # TODO: add an option where you could rescale the variables to match the scale.
        use_legend = kwargs.get("use_legend", False)
        separate_legend = kwargs.get('separate_legend', False)
        if multi_plot != 1:
            assert len(results) == multi_plot
            multi_results = results
        else:
            multi_results = [results]
        for multi_plot_idx,results in enumerate(multi_results):
            if multi_plot_idx == 0:
                figsize = kwargs.get("figsize",self.prefs.determine_figsize(1,multi_plot,use_legend,separate_legend))
                fig = plt.figure(figsize=figsize)
                ax = fig.add_subplot(1,multi_plot,1)
            else:
                ax = fig.add_subplot(1,multi_plot,multi_plot_idx+1)
            stochastic_styling = kwargs.get("stochastic_styling", "mean")
            if stochastic_styling == "mixed":
                stochastic_styling = "mean_traces"
            plot_std = False
            min_values = []
            for idx, result in enumerate(results):
                exp = result["experiment"]
                time_axis,timescale,start,stop,_ = self._get_time_quantities(exp,**kwargs)
                values = self.reduce_data(result,start,stop,kwargs.get("step",1))
                self.color_coding.reset_color_cycle()
                min_values.append(self.get_min_value(values,variables = variables))
                for var in variables:
                    if hasattr(self.models[result["model"]].variables,var) and var in result.keys():
                        col = self.color_coding.get_next_color("var")
                        # Check if we're plotting a deterministic or stochastic variable
                        try: 
                            len(result[var][0])
                            if rescale:
                                if idx == 0:
                                    ref_value = values["mean_"+var][0]
                                else:
                                    for v in values:
                                        if var in v:
                                            values[v] = values[v]/values[v][0]*ref_value
                            plot_std = True
                        except:
                            if rescale:
                                if idx == 0:
                                    ref_value = values[var][0]
                                else:
                                    values[var] = values[var]/values[var][0]*ref_value
                            plot_std = False
                        if not plot_std:
                            plt.plot(time_axis, values[var],color = col,label = "deterministic")
                        else:
                            if "traces" in stochastic_styling:
                                try:
                                    trace_n = int(stochastic_styling[-1])
                                except:
                                    trace_n = len(values[var])
                                for k in range(trace_n):
                                    # Choose from a different list
                                    trace_col = self.color_coding.get_next_color()
                                    plt.plot(time_axis, values[var][k],color = trace_col, alpha = 0.7, label = "trace")
                            if "mean" in stochastic_styling:
                                std = "std_" + var 
                                mean = "mean_"+var
                                plt.plot(time_axis, values[mean],color = col, label = "mean")
                                plt.fill_between(time_axis,values[mean]-values[std],values[mean]+values[std],alpha = 0.2,color = col)          
                plt.xlabel(kwargs.get("xlabel","Time in minutes"))
                plt.ylabel(kwargs.get("ylabel", "Volume"))
            min_value = np.min(min_values)
            self.plot_experiment_quantities(exp,min_value,result["model"],time_axis[-1],timescale)
            if exp.experimental_data and exp_data:
                    self.plot_experimental_data(exp,values,plot_std,timescale)
        if use_legend:
            if not separate_legend:
                plt.legend(bbox_to_anchor = (1,0),loc = "lower left")
                plt.tight_layout()
            else:
                self.plot_separate_legend(axis = ax, filename=filename)
        makedirs(self.prefs.plot_directory[:-1],exist_ok=True)
        fig.savefig(self.prefs.plot_directory + filename)
        plt.show()
    
    def compare_models(self, models = None, exp_names = None, **kwargs):
        #TODO: finish!
        if exp_names is None: exp_to_plot = []
        if models is None:
            # If no choice is given, just take all the models for which we have results saved
            models = [model for model in self.model_results]
        for model in models:
            model_experiments = [result["experiment"].name for result in self.model_results[model]]
            if not model in self.models:
                raise ValueError(f"The selected model is not defined in the simulation: {model}")
            if exp_names is not None:
                for exp in exp_names:
                    if not exp in model_experiments:
                        raise ValueError(f"The selected experiment ({exp}) was not run on the model {model}")
            else:
                if len(exp_to_plot) == 0:
                    exp_to_plot = model_experiments
                else:
                    # Since we're removing stuff it's safer to do it like this
                    tmp = exp_to_plot
                    for exp in tmp:
                        # We only plot the experiments that the models have been all been run with.
                        if exp not in model_experiments:
                            exp_to_plot.remove(exp)
                            warn(f"The experiment {exp} was not in the results for model {model}, and will not be plotted")
        if exp_names is None: exp_names = exp_to_plot
        results = []
        for exp in exp_names:
            # Currently this function is not implemented correctly
            result = self.compose_results(exp,models)
            results.append(result)
        self.plot_results(results, **kwargs)
    
    def compose_results(self,exp,models, save_model = False):
        """
        Take the experiment results from any models and combine them into a composite result.
        This also creates a DummyModel class which contains the variables from all the models and
        other special attributes. This will be continuously overwritten (TODO coud put option to retain it),
        so that you don't risk running out of memory..
        TODO Make it user-ready..
        """
        for model in models:
            model_experiments = [result["experiment"].name for result in self.model_results[model]]
            result = self.model_results[model][model_experiments.index(exp)]
            new_keys = [model + name for name in result.keys()]
        
    
    def plot_comparison(self, results = None, models = None, exp_names = None, **kwargs):
        """
        Compare different results. 
        Params:
        -------
                results : list, optional
                        A list of the results that should be compared
                        This should not be given in input if also a list of models is given.
                models: list, optional
                        A list of the models whose results you want to plot (note that latest results for each of the models
                        in the simulation are saved automatically, so you don't need to also give a list of results connected to
                        these models)
                exp_names: list(str), optional
                        A list of the experiments that should be plotted, if not given, all the results that are given or that have been saved
                        by the simulation will be used.
        """
        # TODO: Make sure that the axes have the same scale so they're actually comparable!
        if results is not None and models is not None:
            raise ValueError("You gave both a list of results and a list of models! Only one of these should be given as input")
        if results is None and models is None:
            results = deepcopy(self.model_results)
        if models is not None and results is None:
            results = {}
            for model in models:
                results[model] = [self.model_results[model][i].copy() for i in range(len(self.model_results[model]))]
        elif results is not None and isinstance(results,list):
            results = {}
            for result in results:
                results[result["model"]] = result
        result_list = self.sort_results_by_exp(results, exp_names = exp_names)
        # Giving exp_names at this point is useless, but I guess it doesn't hurt
        self.plot_results(result_list,exp_names,n_cols = len(results.keys()), **kwargs)
    
    def sort_results_by_exp(self,res_dict, exp_names = None):
        # Not sure if this is necessary but I guess it's a good precaution
        res_dict = res_dict.copy()
        res_list = []
        if exp_names is None:
            exp_names = []
            for m in res_dict.keys():
                names = [res_dict[m][i]["name"] for i in range(len(res_dict[m]))]
                if len(names) > len(exp_names):
                    exp_names = names
        for name in exp_names:
            for model in res_dict.keys():
                # Take the results from each of the models
                results = res_dict[model]
                names = [results[i]["name"] for i in range(len(results))]
                if name in names:
                    idx = names.index(name)
                    res = results.pop(idx)
                    res_list.append(res)
        return res_list 

    def plot_selected_points(self,results,variables,selection_idx, 
                        event_type = "stim", event_idx = 60, x_data = None,
                        reference_values = None, reference_data = None, multi_plot = 1,
                        save = True, filename = "img.pdf", **kwargs):
        # TODO: add possibility to plot from multiple results (i.e. a list or dictionary of results like above)
        # TODO: general thing, implement a figure class which makes it easier to manipulate images without the need to:
        # 1. Always call these plotting functions with 100 arguments
        # 2. Always re-plot everything and keep adding functionalities which would be shared among all plots you can do with this package.
        
        if not isinstance(results,list):
            if isinstance(results,dict):
                results = [results]
            else:
                raise ValueError("The results you gave me are not in the correct format, they should either be a list of dicts or a single dict")
        if multi_plot != 1: 
            assert len(results) == multi_plot
            if reference_data is not None:
                assert len(reference_data) == multi_plot
                multi_data = reference_data
            else:
                multi_data = [None for k in range(multi_plot)]
            if reference_values is not None:
                assert len(reference_values) == multi_plot
                multi_values = reference_values
            else:
                multi_values = [None for k in range(multi_plot)]
            multi_results = results
        else:
            multi_results = [results]
            multi_data = [reference_data]
            multi_values = [reference_values]
        for multi_plot_idx,[results,reference_data,reference_values] in enumerate(zip(multi_results,multi_data,multi_values)):
            if not isinstance(variables,list):
                variables = [variables]
            for result in results:
                for var in variables:
                    if not var in result.keys():
                        raise KeyError("The name of the variable to plot should match the one given in the results dictionary! "
                        f"But in the result named {result['name']}, the selected variable is not present.")
            if not (isinstance(selection_idx,str) or isinstance(selection_idx,int)):
                raise ValueError("The selection index should be given as a string or an int!")

            if isinstance(selection_idx,str):
                if not selection_idx in self.prefs.point_selection:
                    raise NotImplementedError(f"The selection index {selection_idx} is currently not supported. "
                    f"The supported string indices are: {self.prefs.point_selection}")
                # These quantities are useful later when looking for the points to plot
                multiplier = 1 if not "last" in selection_idx else 0
                interval_choice = -1 if "after" in selection_idx else 0
                # This is the only case we can cover here without looking at the results
                if selection_idx == "last":
                    selection_idx = -1
            # eventcting the data from the results
            time_axis = []
            points = {varname: [] for varname in variables}
            titles = []
            for result in results:
                dt = result["experiment"].dt
                if isinstance(selection_idx,str):
                    # Given multiple event types we order them temporally 
                    events = self.sort_events_in_t(result["experiment"],event_type,interval_choice)
                    if len(events)> 1: # I assume if you have multiple events you want one plot per experiment
                        # Another assumption, the time axis is shared among different variables
                        time_axis.append([events[i][interval_choice] - 1 - multiplier*event_idx for i in range(len(events))])
                        dt = result["experiment"].dt
                        for varname in variables:
                            points[varname].append([result[varname][int(t/dt)] for t in time_axis[-1]])
                        titles.append(result["model"] + ":" + result["experiment"].title)
                    else:
                        time_axis.append(events[0][interval_choice]-1-multiplier*event_idx)
                        for varname in variables:
                            points[varname].append(result[varname][int(time_axis[-1]/dt)])
                        titles.append(result["model"] + ":" + result["experiment"].title)
                    x_is_time = True
                else:
                    # Assuming you just want the experiments to be numbered
                    # TODO: could give an option to specify what to put on the time axis
                    time_axis.append(len(time_axis))
                    x_is_time = False
                    for varname in variables:
                        points[varname].append(result[varname][selection_idx])
                    titles.append(f"{result['model']}:{kwargs.get('title', result['experiment'].title)}")
            # This fixes a bug related to plotting only one experiment
            if len(time_axis) == 1:
                for varname in variables:
                    points[varname] = points[varname].pop(0)
                time_axis = time_axis.pop(0)
            
            n_plots = 1 if not isinstance(points[variables[0]][0],list) else len(results)
            self.logger.info(f"Making {n_plots} plots")
            # TODO this is not optimal, also, currently not supporting multiple x_data arrays
            if x_data is not None: # Override the time axis, but only if they're compatible
                if n_plots> 1:
                    for idx,time in enumerate(time_axis):
                        if len(x_data) != len(time):
                            warn(f"The x_data you gave is not compatible with the number of points in the plot, "
                            f"currently not using it. given_data {len(x_data)} != {len(time)} selected points")
                        else:
                            time_axis[idx] = x_data
                            x_is_time = False
                else: 
                    if len(x_data) != len(time_axis):
                        warn("The x_data you gave is not compatible with the number of points in the plot, "
                            f"currently not using it. given_data {len(x_data)} != {len(time_axis)} selected points")
                    else:
                        time_axis = x_data
                        x_is_time = False
            if reference_values is not None:
                # Making sure the formatting is done correctly
                if isinstance(reference_values,list):
                    if len(reference_values) % n_plots == 0 and n_plots > 1:
                        if not isinstance(reference_values[0],list):
                            reference_values = [[reference_values[i]]for i in range(len(reference_values))]
                    else:
                        if isinstance(reference_values[0],list) or isinstance(reference_values[0],np.ndarray):
                            raise ValueError("The reference values you gave are incompatible with the number of plots, make sure you either give"
                            " a list of values (or single value) for each plot, or a list of values that should be shared among plots.")
                        else:
                            reference_values = [reference_values for k in range(n_plots)]
                else:
                    reference_values = [[reference_values] for k in range(n_plots)]
            if reference_data is not None:
                if n_plots > 1:
                    if len(reference_data) != n_plots:
                        raise ValueError("The reference data is not compatible with the requested plots.")
                    assert all([len(reference_data[i]) == len(time_axis[i]) for i in range(len(time_axis))])
                else:
                    assert len(reference_data) == len(time_axis)
                    reference_data = [reference_data]
            # Convert time to minutes
            if x_is_time:
                time_unit = get_time_unit_in_seconds(kwargs.get("time_unit","m"))
                if n_plots> 1:
                    time_axis = [[int(t/time_unit) for t in time_axis[i]] for i in range(len(time_axis))]
                else:
                    time_axis = [int(t/time_unit) for t in time_axis]
            if multi_plot_idx == 0:
                plot_cols = kwargs.get('n_cols',min(n_plots,4))*multi_plot
                plot_rows = ceil(n_plots/plot_cols)
                use_legend = kwargs.get("use_legend", True)
                separate_legend = kwargs.get('separate_legend', False)
                figsize = kwargs.get('figsize',self.prefs.determine_figsize(plot_rows,plot_cols,use_legend,separate_legend))
                self.logger.debug("Figsize",figsize)
                fontsize = kwargs.get('fontsize',12)
                plt.rcParams['font.size'] = fontsize
                fig = plt.figure(figsize=figsize)
                ax0 = fig.add_subplot(1,multi_plot,1)
                if multi_plot == 1:
                    ax0.set_axis_off()
            else:
                ax = fig.add_subplot(1, multi_plot, multi_plot_idx+1,sharey = ax0)
            axes = []
            for idx in range(n_plots):
                if reference_data is not None:
                    t_axis = time_axis[idx] if n_plots > 1 else time_axis
                    labels = kwargs.get("ref_data_labels","ref")
                    label = labels[idx] if n_plots > 1 else labels
                    plt.plot(t_axis,reference_data[idx],label = label,lw = 3)
                if n_plots != 1:
                    axes.append(fig.add_subplot(ceil(n_plots/plot_cols),plot_cols if n_plots >= plot_cols else n_plots,idx+1,
                    sharey =axes[0] if idx > 0 else None))
                for varname in variables:
                    t_axis = time_axis[idx] if n_plots > 1 else time_axis
                    p = points[varname][idx] if n_plots >1 else points[varname]
                    var_label = self.determine_label(varname)
                    plot_style = kwargs.get('plot_style','line')
                    if plot_style == 'line':
                        plt.plot(t_axis,p,label = var_label, color = self.color_coding[varname],lw = 3)
                        # # TODO Delete this!!
                        # if varname == "V_tot":
                        #     plt.plot([0,0],[0.6,2],lw = 3,ls = "--", c = "brown",label = r"$LFS$")
                        # l1 = r"15'" if varname == "V_tot" else None
                        # l2 = r"45'" if varname == "V_tot" else None
                        # plt.scatter(t_axis[12],p[12], lw = 2, marker = 'o',c = "teal", label = l1,zorder = 10)
                        # plt.scatter(t_axis[18],p[18], lw = 2, marker = 'o',c = "forestgreen", label = l2, zorder = 10)
                    else:
                        plt.plot(t_axis,p,label = var_label,ls = None,marker = 'o',color = self.color_coding[varname])
                if reference_values is not None:
                    default_labels = [f"ref {j}" for j in range(len(reference_values))]
                    for i,val in enumerate(reference_values[idx]):
                        l = kwargs.get('ref_labels',default_labels)[i]
                        plt.plot([time_axis[0],time_axis[-1]],[val,val],label = l,ls = "--",lw = 3)
                plt.xlabel(kwargs.get("xlabel","iterations"))
                plt.ylabel(kwargs.get('ylabel','ratio of baseline'))
                plt.title(titles[idx])
                # yticks = plt.yticks()[0]
                #TODO delete!
                # if multi_plot_idx != 1:
                #     time = 15 if "EC" in titles[idx] else 45
                #     plt.plot([time,time],[1.5,3.1],lw = 3,ls = "--", c = "brown",label = r"$\phi(t)= 1$")
                # else:
                #     time = 15 if "EC" in titles[idx] else 45
                #     plt.plot([time,time],[1.5,3.1],lw = 3,ls = "--", c = "brown",label = r"$\phi(t)= 1$") 
                # plt.scatter(time_axis[2],points["Vpsd"][2], label = "5 min", c = "brown", lw = 3)
                # plt.scatter(time_axis[7],points["Vpsd"][7], label = "15 min", c = "teal", lw = 3)
                # plt.xlim(right = 47)
        if use_legend:
            if not separate_legend:
                plt.legend(bbox_to_anchor = (1.0,0), loc = "lower left")
                plt.tight_layout()
            else:
                self.plot_separate_legend(axis = ax,filename = filename)
        makedirs(self.prefs.plot_directory[:-1],exist_ok=True)
        fig.savefig(self.prefs.plot_directory + filename)
        plt.show()
    
    def determine_label(self,varname):
        if not hasattr(self.model,"labels") or not varname in self.model.labels:
            return varname
        return self.model.labels[varname]
    
    def determine_varname(self,label):
        if not hasattr(self.model,"labels") or not label in self.model.labels.values():
            return label
        return list(self.model.labels.keys())[list(self.model.labels.values()).index(label)]


    def sort_events_in_t(self,exp,event_type,start_end):
        """
        Sorts a list of events from an Experiment with respect to their 
        start or end time (given by start_end (0 or -1)). Note that events can have
        different lengths and can be superimposed, so that's why it's important to know
        how to sort them.
        """
        start_times = []
        events = []
        for event in event_type:
            for interval in getattr(exp,event):
                events.append(interval)
                start_times.append(interval[start_end])
        idx = np.argsort(np.asarray(start_times))
        return np.array(events)[idx]

    def copy(self):
        """
        Make a copy of the current class and return it
        """
        namespace = vars(self).copy()
        S = Simulation(namespace)
        return S