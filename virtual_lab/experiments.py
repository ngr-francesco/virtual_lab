from warnings import warn
from virtual_lab.settings import prefs
import numpy as np

class Experiment:
    def __init__(self,name: str, event_intervals: dict(), exp_data = None, **kwargs):
        self.name = name
        self.title = kwargs.get('title',name)
        self.quantities = {}
        self.experimental_data = exp_data
        for key,value in event_intervals.items():
            setattr(self,key,[[onset,onset+duration] for onset,duration in value])
            self.quantities[key] = getattr(self,key)
        self.dt = kwargs.get("dt",1)
        self.T = kwargs.get("T",prefs.exp_T)
    
    steps = property(lambda self: int(self.T/self.dt))
    
    def __call__(self):
        return self.quantities
    
    def __str__(self):
        return self.name
    
    def copy(self):
        event_intervals = {}
        for name, q in self.quantities.items():
            new_q = []
            for ton,toff in q:
                duration = toff-ton
                new_q.append([ton,duration])
            event_intervals[name] = new_q

        return Experiment(self.name,event_intervals,T = self.T,dt = self.dt, exp_data = self.experimental_data)

    def edit_quantity(self,name,values):
        if name in self.quantities:
            self.quantities[name] = values
            setattr(self,name,values)
    
    def add_event(self,name,intervals):
        setattr(self,name,[[onset,onset+duration] for onset,duration in intervals])
        self.quantities[name] = getattr(self,name)


class Experiments:
    def __init__(self,experiments = None):
        """
        Just a simple container for various experiments
        """
        self.experiments = []
        self._exp_names = []
        if experiments is not None:
            self.add(experiments)
        
    exp_names = property(lambda self: self._exp_names)

    def add(self, experiments, idx = None):
        try:
            len(experiments)
        except TypeError:
            experiments = [experiments]
        for experiment in experiments:
            if not isinstance(experiment,Experiment):
                raise TypeError("The experiments should be "
                "instances of the Experiment class, instead I found a "
                f"{type(experiment)} type")
            if idx is None:
                self.experiments.append(experiment)
                self._exp_names.append(experiment.name)
            else:
                if not idx in range(len(self.experiments)):
                    warn(f"Index {idx} is not valid for this class containing"
                    " {len(self.experiments)} experiments. Will simply append at the end") 
                self.experiments.insert(idx,experiment)
                self._exp_names.append(experiment.name)

    def __iter__(self):
        return iter(self.experiments)
    
    def __getitem__(self,idx):
        return self.experiments[idx]
    
    def index(self,name):
        return self.exp_names.index(name)

    def __len__(self):
        return len(self.experiments)

    def get_experiment(self, _name: str):
        if _name in self.exp_names:
            for experiment in self.experiments:
                if experiment.name == _name:
                    return experiment
        raise Warning("Experiment wasn't found. Make sure you are using the " 
                    "correct name")

    def __str__(self):
        return [str(experiment) for experiment in self.experiments]
    
    def __len__(self):
        return len(self.experiments)

    def copy(self):
        return Experiments(experiments= [exp.copy() for exp in self.experiments])
    
    def add_event(self,name,intervals, concurrence = None):
        for exp in self.experiments:
            if concurrence is None:
                warn("You're currently adding an event to all your experiments, this is rarely what you really want to do!"
                "\nIf you want to add an event to a single experiment you can modify the experiment directly by using EXP_NAME.add()!")
                exp.add_event(name,intervals)
            else:
                if (isinstance(intervals[0], float) or isinstance(intervals[0],np.float32)) or (isinstance(intervals[0],int) or isinstance(intervals[0],np.int32)):
                    if len(getattr(exp,concurrence))>0:
                        exp.add_event(name,[[i[0] + intervals[0],intervals[1]] for i in getattr(exp,concurrence)])
                    else:
                        exp.add_event(name,[])
                else:
                    raise NotImplementedError("If an event is concurring with another and you wish to add it, use the format [onset,duration] (no nested lists!)")

    







