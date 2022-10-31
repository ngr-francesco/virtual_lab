import matplotlib
import numpy as np
import matplotlib.pyplot as plt
from math import ceil
from warnings import warn
from os import makedirs
from copy import deepcopy
from itertools import chain
from time import perf_counter
from virtual_lab.experiments import Experiments, Experiment
from virtual_lab.settings import prefs
from virtual_lab.model import Model

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
        if model is not None:
            self.model = model
            self.models = {model.name: model}
        else:
            self.models = {}
            print("Initializing a simulation without a given model, a model should be added before running any experiment"
            " by calling Simulation.add_model()")
        self.model_results = {}
        self.prefs = kwargs.get("settings",prefs)

    def add_model(self,model, switching = True):
        self.models[model.name] = model
        if switching:
            print(f"Switching to model {model.name}")
            self.model = model
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
            print("stochastic process run..")
        if self.model.termalization_step is not None:
            term_func = self.model.termalization_step
        else:
            term_func = self.model.single_step
        for t in range(termalization_time):
            term_func(t)
        print("termalization process run..")
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
        constants_changed = False
        experiment_constants = None
        custom_const = None
        start_time = perf_counter()
        for i, exp in enumerate(experiments):
            # Make it simple to change nucleation rates for single experiments
            results[i]["name"] = exp.title
            results[i]["experiment"] = exp
            results[i]["model"] = self.model.name
            self.model.dt = exp.dt # TODO: Better management of the dt
            # I change the nf values but keep track if they are different 
            # because if they aren't we don't need to redo termalization
            if exp.has_custom_constants: 
                custom_const = exp.get_custom_constants()
            constants_changed = experiment_constants != custom_const
            experiment_constants = custom_const
            if termalize and constants_changed:
                termalized = False
            self.model.prepare_constants(exp,exclude_stochastic = stochastic)
            # In case we're running a termalization process we only want to run this once
            # because then re-initialization is automatically taken care of inside of model.reset()
            if not stochastic or not termalized:
                self.model.initialise_variables(exp,stochastic)
            for k in range(n_of_runs):
                
                # INITIALIZE ALL OF THE QUANTITIES FOR THE SIMULATION
                # If we want to allow the system to termalize on its own
                if stochastic:
                    self.model.run_stochastic_processes()
                if termalize:
                    # If the system doesn't change between experiments we only run this once!
                    if not termalized:
                        print("Termalizing...")
                        self.termalize_system(termalization_time,exp,stochastic, record_termalization)
                        termalized = True
                        print("Termalized")
                # Simulation Loop
                # TODO check if this is considerably faster if done within the model
                for t in range(exp.steps):
                    self.model.single_step(t)
                # Save the data
                
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
        print(f"Simulating {len(experiments)} experiments took: {duration:.3f} s")
        return results
    
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
        figsize = (kwargs.get('figsize',(8,6)))
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


    def plot_volumes(self, results = None, exp_names = None, 
                    time_interval = None, filename = "img.pdf", 
                    time_in_min = True, step = 1, show_equations = True, **kwargs):
        """
            Plot the volumes computed in the experiments.
        Parameters:
        ----------
            results (list(dict)): containing the results of the experiments in a dictionary,
                        generated by the method simulation_routine()
            exp_names (list(int)): the list of the experiments to be plotted
            time_interval (tuple,optional): define the time interval to be plotted, defaults to None
            name (string,optional): the name to give to the output image, defaults to 'img.pdf'
            time_in_min (bool,optional): whether the time axis should be given in minutes or hours, defaults to True
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
        print(f"Plotting {len(results) if exp_names is None else counter} experiments")
        plot_cols = kwargs.get("n_cols",4)
        figsize = kwargs.get('figsize',
                            (min((len(results) if exp_names is None else counter)*10,20),
                            min(ceil((len(results) if exp_names is None else counter)/plot_cols)*7,104)))
        fontsize = kwargs.get('fontsize',12)
        use_title = kwargs.get('use_title',True)
        variables = kwargs.get('variables',None)
        lw = kwargs.get('lw',3)
        use_legend = kwargs.get('legend', True)
        fig = plt.figure(1, figsize=figsize)
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
        
        quantities_for_legend = 0
        plot_idx = 0
        rows = ceil(len(results)/plot_cols)
        cols = (plot_cols if len(results) >= plot_cols else len(results))
        rows = rows+1 if show_equations else rows
        start_index = 1 + cols if show_equations else 1
        if show_equations:
            models = list(set([result["model"] for result in results]))
            self.model_equations(models,fig,rows)
        for result in results:
            if result["name"] in exp_names:
                idx = exp_names.index(result["name"])
                start = time_interval[idx][0] if time_interval is not None else 0
                dt = result["experiment"].dt if step is None else step*result["experiment"].dt
                stop = time_interval[idx][1] if time_interval is not None else int(result["experiment"].T) 
                timescale = 60 if time_in_min else 3600
                time_axis = np.arange(start,stop,dt)/timescale
                values = self.reduce_data(result,start,stop,step)
                # Check if we ran simulations with statistics
                if  "mean_Vd" in values.keys():
                    plot_std = True
                else:
                    plot_std = False
                    #print(len(Vd_t),len(Vs_t))

                ax = fig.add_subplot(rows,cols,plot_idx+start_index)

                if use_title:
                    plt.title(result["model"] + ": " + result["name"])
                
                plt.plot([start,stop/timescale],[1,1] ,color='0', ls=":")
                if variables is None:
                    model_variables = self.models[result["model"]].variables.varnames
                else:
                    model_variables = variables
                for name in model_variables:
                    if name in values.keys():
                        if plot_std:
                            plt.plot(time_axis, values["mean_" + name], label = name, lw = lw)
                            plt.fill_between(time_axis,values["mean_" + name] - values["std_"+name],
                                                    values["mean_" + name]+ values["std_" + name],alpha = 0.3)
                        else:
                            plt.plot(time_axis,values[name],label = name,lw = lw)
                if result["experiment"].experimental_data is not None:
                    colors = ['k','b','g'] # Assuming you don't want to plot more than 3 experiments on top of your results
                    data_dict = result["experiment"].experimental_data
                    try:
                        renorm_value = values["mean_" + data_dict["renorm_var"]][0] if plot_std else values[data_dict["renorm_var"]][0]
                    except KeyError:
                        renorm_value = 1
                    # TODO: This is a bit terrible
                    plot_dict = data_dict.copy()
                    plot_dict.pop("renorm_var")
                    for i,data in enumerate(plot_dict):
                        # TODO: This is also highly incorrect and should happen in other places
                        y_data = np.array(data_dict[data]["y"])*renorm_value
                        plt.plot(data_dict[data]["x"],y_data,label = data,lw = 2,marker = 'x',markersize = 10,ls='--',color = colors[i])
                        
                #plt.plot(np.arange(0,T+dt,dt)/3600, (Vs_t+Vd_t-PSD_95)/2 +shift, label=r'Tag', ls='-.', alpha =0.2)
                #plt.plot([0,T/3600],[shift,shift] ,color='0', ls=":")

                #plt.plot(np.arange(0,T+dt,dt)/3600,Vd_t /(Vs_t+Vd_t)-eql+0.5, ls=":", label=r'Diff from Eql.')
                #plt.plot([0,T/3600],[0.5,0.5] ,color='0', ls=":", alpha=0.2)

                # Plotting the quantities used in our experiment
                quantities = result["experiment"]()
                # This counter is useful for knowing which of the experiments should determine the legend (the one with the most quantities being plotted)
                # and also to know in which positions to plot the quantities so they don't overlap vertically
                counter = 0
                for q in quantities.keys():
                    if quantities[q] is not None and len(quantities[q]):
                        counter += 1
                positions = np.arange(-0.7,-0.1,0.6/len(quantities.keys()))
                # For now I just use a long list which should contain enough colors for all intents and purposes
                colors = ['royalblue','mediumaquamarine','khaki','gainsboro','darkviolet',
                'xkcd:green gray','xkcd:dark gold','xkcd:hospital green','xkcd:rose red']

                # Only take a number of elements equal to the number of quantities
                colors = colors[:len(quantities.keys())]
                # Make a list of the quantities needed for the model, we only want to plot those.
                dep = list(self.models[result["model"]].quantity_dependencies().values())
                # Flatten the list of the keys contained in the values of the dictionary returned by quantity_dependencies()
                used_quantities = list(chain(*[list(dep[i].keys()) for i in range(len(dep))]))
                for q,pos,col in zip(quantities.keys(),positions,colors):
                    if q in used_quantities:
                        plt.plot([0,time_axis[-1]],[pos,pos], color = col, ls = ":", alpha = 0.5)
                        if quantities[q] is not None and len(quantities[q]):
                            for ton,toff in quantities[q]:
                                toff = min(toff,stop)
                                plt.plot([ton/timescale,toff/timescale], [pos,pos], color = col,lw = 6)
                            plt.plot([ton/timescale,toff/timescale], [pos,pos],label = q,color = col, lw = 6)
                        

                plt.xlabel("t in min",fontsize=fontsize)
                plt.ylabel(r"% of Volume",fontsize=fontsize)
                if use_legend:
                    use_this_exp = True if (counter > quantities_for_legend) else False
                    quantities_for_legend = counter if use_this_exp else quantities_for_legend
                    if use_this_exp:
                        legend_axes = ax
                plot_idx += 1

        # Esthetic styling
        plt.xticks(fontsize=fontsize)
        plt.yticks(fontsize=fontsize)
        plt.gca().spines['top'].set_visible(False)
        plt.gca().spines['right'].set_visible(False)
        # Save figure in the plots directory
        if use_legend:
            handles,labels = legend_axes.get_legend_handles_labels()
            plt.legend(handles,labels, bbox_to_anchor = (1,0), loc= 'lower left', ncol=kwargs.get('legend_cols', ceil(len(model_variables)/4)), fontsize=fontsize)
        makedirs(self.prefs.plot_directory[:-1],exist_ok=True)
        fig.savefig(self.prefs.plot_directory + filename)
        plt.show()
    
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
        self.plot_volumes(result_list,exp_names,n_cols = len(results.keys()), **kwargs)
    
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
                        reference_values = None, save = True, filename = "img.pdf", **kwargs):
        # TODO: add possibility to plot from multiple results (i.e. a list or dictionary of results like above)
        if not isinstance(results,list):
            if isinstance(results,dict):
                results = [results]
            else:
                raise ValueError("The results you gave me are not in the correct format, they should either be a list of dicts or a single dict")
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
        # Extracting the data from the results
        time_axis = []
        points = {varname: [] for varname in variables}
        titles = []
        for result in results:
            if isinstance(selection_idx,str):
                # Given multiple event types we order them temporally 
                events = self.sort_events_in_t(result["experiment"],event_type,interval_choice)
                if len(events)> 1: # I assume if you have multiple events you want one plot per experiment
                    # Another assumption, the time axis is shared among different variables
                    time_axis.append([events[i][interval_choice] - 1 - multiplier*event_idx for i in range(len(events))])
                    dt = result["experiment"].dt
                    for varname in variables:
                        points[varname].append([result[varname][int(t/dt)] for t in time_axis[-1]])
                    titles.append(result["model"] + ":" + result["name"])
                else:
                    time_axis.append(events[0][interval_choice]-1-multiplier*event_idx)
                    for varname in variables:
                        points[varname].append(result[varname][int(time_axis[-1]/dt)])
                    titles.append(result["model"] + ":" + result["name"])
            else:
                # Assuming you just want the experiments to be numbered
                # TODO: could give an option to specify what to put on the time axis
                time_axis.append(len(time_axis))
                for varname in variables:
                    points[varname].append(result[varname][selection_idx])
                titles.append(f"{result['model']}:{result['name']}")
        if x_data is not None: # Override the time axis, but only if they're compatible
            if len(x_data) != len(time_axis):
                warn("The x_data you gave is not compatible with the number of points in the plot, "
                "currently not using it.")
            else:
                if not isinstance(x_data[0],list) or isinstance(x_data[0],np.ndarray):
                    time_axis = x_data
                else:
                    raise NotImplementedError("Multiple x_data for these plots is currently not supported.")
        
        n_plots = 1 if not isinstance(points[variables[0]][0],list) else len(results)
        print(f"Making {n_plots} plots")
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
        # Convert time to minutes 
        if kwargs.get("time_in_m",True):
            if n_plots> 1:
                time_axis = [[int(t/60) for t in time_axis[i]] for i in range(len(time_axis))]
            else:
                time_axis = [int(t/60) for t in time_axis]

        plot_cols = kwargs.get('n_cols',3)
        figsize = kwargs.get('figsize',(
                            (min(n_plots*10,20)),
                            min(ceil(n_plots/plot_cols)*6,104)))
        fontsize = kwargs.get('fontsize',12)
        fig = plt.figure(figsize=figsize)
        for idx in range(n_plots):
            plt.subplot(ceil(n_plots/plot_cols),plot_cols if n_plots >= plot_cols else n_plots,idx+1)
            for varname in variables:
                plt.plot(time_axis[idx] if n_plots > 1 else time_axis,points[varname][idx] if n_plots >1 else points[varname],label = varname)
            if reference_values is not None:
                for val in reference_values[idx]:
                    plt.plot([time_axis[0],time_axis[-1]],[val,val],label = kwargs.get('ref_label',"reference"))
            plt.xlabel("Time in m")
            plt.ylabel(kwargs.get('ylabel','% of baseline'))
            plt.legend(fontsize = fontsize)
            plt.title(titles[idx]) 
        if save:
            makedirs(self.prefs.plot_directory[:-1],exist_ok=True)
            plt.savefig(self.prefs.plot_directory + filename)
        plt.show()
    
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