from distutils.log import warn
import numpy as np
from virtual_lab.experiments import Experiment
# TODO: Maybe it's possible to avoid having to write every time self.variables.variable and you could
# set a proxy reference to that variable where you could just say self.variable
class Variables:
    """
    A class managing variables.
    Variables should be given in the form of a dictionary with "name": init_value
    """
    def __init__(self, variables, record = True):
        for key,value in variables.items():
            setattr(self,key,value)
        self.init_values = variables.copy()
        self.varnames = [var for var in variables]
        self._to_record = {name: record for name in self.varnames}
        self.recorded_variables = [name for name in self.varnames if record]
        self._is_recording = record
        # For now we either record all or none
        if self.is_recording:
            self.recorded_values = {name : [] for name in self.varnames}
    
    to_record = property(lambda self: self._to_record)
    
    def __iter__(self):
        return iter([var for var in self.varnames])
    
    def __len__(self):
        return len(self.varnames)

    @property
    def is_recording(self):
        return self._is_recording
    
    @is_recording.setter
    def is_recording(self,record:bool):
        self._is_recording = record
        # Only change the variables which were previously being recorded!
        self._to_record.update({name: record for name in self.recorded_variables})
    
    def get_values(self):
        return [getattr(self,name) for name in self.varnames]

    def set_values(self, values):
        for name,value in values.items():
            if name in self.varnames:
                setattr(self,name,value)
                if self.is_recording and self._to_record[name]:
                    self.recorded_values[name].append(value)

    def diff_timestep(self,values,dt):
        for name,value in values.items():
            new_value = getattr(self,name) + dt*value
            setattr(self,name,new_value)
            if self.is_recording and self._to_record[name]:
                self.recorded_values[name].append(new_value)

    def update_recorded_variables(self,values):
        if isinstance(values,str):
            if values == 'all':
                self.recorded_variables = [var for var in self.varnames]
                self.is_recording = True
            elif values == 'none' or values == 'None':
                self.recorded_variables = []
                self.is_recording = True
        elif isinstance(values,dict):
            to_pop = []
            for name in values:
                if not name in self.varnames:
                    warn(f"An incorrect variable name was give, this variable will be ignored: {name}")
                    to_pop.append(name)
            for name in to_pop:
                values.pop(name)
            self._to_record.update(values)
            self.recorded_variables = [name for name in self.varnames if self._to_record[name] == True]
            self.is_recording = True if len(self.recorded_variables) else False
        # Here we assume the user gave a list of variables to record
        elif isinstance(values,list):
            for name in values:
                if name in self.varnames:
                    self._to_record[name] = True
                    if not name in self.recorded_variables:
                        self.recorded_variables.append(name)
                else:
                    warn(f"An incorrect variable name was given, this variable will be ignored: {name}") 



class Model:
    def __init__(self,name:str, model_variables:dict, const:dict = {}, labels = None, record = True):
        # TODO: Implement a dictionary system for stochastic variables, right now it only works for a single one!
        self.const = const
        self.name = name
        # Set up the variables
        self.variables = Variables(model_variables, record)
        # Set up the values of other relevant quantities
        for key,value in const.items():
            setattr(self,key,value)
        # Make a dict containing all the equations of the model (must match the names of the variables!)
        self.model_equations = self.equations_dict()
        self.diff_equations = self.diff_equations_dict()
        self.stochastic_variables = self.stochastic_variables_dict()
        self.labels = labels if labels is not None else {}
    
    # Some proxy methods (not sure if this is legit but I guess it's okay)
    @property
    def is_recording(self):
        return self.variables.is_recording
    @is_recording.setter
    def is_recording(self,record):
        self.variables.is_recording = record

    def record_variables(self,values):
        self.variables.update_recorded_variables(values)

    def stop_recording_variables(self,variables):
        # In this case using a dictionary is a bit useless, since we would put them to false anyways
        # by doing this we're removing human error from the equation
        if isinstance(variables,dict):
            variables = list(variables.keys())
        elif isinstance(variables,str):
            variables = [variables]
        vars_dict = {}
        for var in variables:
            vars_dict[var] = False
        self.variables.update_recorded_variables(vars_dict)

    def __repr__(self):
        # TODO: finish implementation
        _str = self.name
        return _str
    
    def update_constants(self,const):
        for key,value in const.items():
            if hasattr(self,key):
                setattr(self,key,value)
            else:
                warn(f"The constant {key} was not part of the model so it won't be updated.\n"
                "If you want to add constants to your model use the method add_constants()")
    
    def add_constants(self,const):
        for key,value in const.items():
            if not hasattr(self,key):
                setattr(self,key,value)
            else:
                warn(f"The constant {key} was already present in the model and it won't be added again.\n"
                "If you wanted to update its value, use the method update_constants() instead.")

    def reset(self):
        # Reset the variable values to their initial values
        being_recorded = self.variables.to_record
        self.variables = Variables(self.variables.init_values, record=self.variables.is_recording)
        self.record_variables(being_recorded)
    
    def update_variables_init_values(self):
        # Used for when we termalize the system and want to keep these as baseline values
        new_init_values = {}
        for var in self.variables.varnames:
            new_init_values[var] = getattr(self.variables,var)
        self.variables.init_values = new_init_values
    
    def set_initial_values(self,stochastic):
        warn("Currently the method 'set_initial_values' is not implemented, this implies"
             " that all the variable and constant initial values were given through the const argument at "
             "initialization of the Model class.")
    
    def initialise_variables(self,exp,stochastic):
        if stochastic:
            self.prepare_constants(exp,termalizing = True,exclude_stochastic=stochastic)
            if stochastic:
                self.run_stochastic_processes()
        # We initialise the model here by using the values obtained from the "basal" stochastic process
        self.set_initial_values(stochastic)
        self.update_variables_init_values()
        if stochastic:
            self.prepare_constants(exp,exclude_stochastic=stochastic)
        
    
    def prepare_constants(self, exp: Experiment, termalizing = False, exclude_stochastic = False):
        const_dict = self.quantity_dependencies()
        self.initialise_constants(const_dict,exp)
        if not termalizing:
            for name,c_values in const_dict.items():
                # If we don't want to prepare the stochastic variables skip these
                # This is useful in the case of a model which can be both stochastic or deterministic
                # To avoid the need of giving different names to the stochastic and non-stochastic versions of the same variable
                if name in self.stochastic_variables.keys() and exclude_stochastic:
                    continue
                _const = getattr(self,name)
                for s in c_values.keys():
                    # If the experiment doesn't involve this quantity, we skip it
                    if not hasattr(exp,s):
                        continue
                    for ton,toff in getattr(exp,s):
                        _const[int(ton/exp.dt):int(toff/exp.dt)] = c_values[s]
                    setattr(self,name,_const)
        
                    
    def initialise_constants(self,const_dict,exp):
        for name in const_dict:
            # Give priority to the values defined within the experiment
            if hasattr(exp,name + '_0'):
                setattr(self,name,np.ones(exp.steps)*getattr(exp,name+'_0'))
            elif hasattr(self,name + '_0'):
                setattr(self,name,np.ones(exp.steps)*getattr(self,name+'_0'))
            else:
                setattr(self,name,np.zeros(exp.steps))
    
    def run_stochastic_processes(self):
        for varname, (func,args) in self.stochastic_variables.items():
            arguments = []
            # Get the numerical values for each argument of the stochastic process
            for arg in args:
                arguments.append(getattr(self,arg))
            value = func(arguments)
            setattr(self,varname,value)

    def quantity_dependencies(self):
        raise NotImplementedError
    
    def stochastic_variables_dict(self):
        """
        Since stochasticity is not necessarily needed, we just return None and if needed the user needs to implement it
        """
        return {}
    
    def single_step(self,t):
        output = {}
        for var in self.model_equations:
            output[var] = self.model_equations[var](t)
        self.variables.set_values(output)
        output.clear()
        for var in self.diff_equations:
            output[var] = self.diff_equations[var](t)
        self.variables.diff_timestep(output,self.dt)

    def update_constants(self, const):
        for key,value in const.items():
            setattr(self,key,value)
    
    def equations_dict(self,variables):
        raise NotImplementedError
    
    def diff_equations_dict(self,variables):
        raise NotImplementedError


