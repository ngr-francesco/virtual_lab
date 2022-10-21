from warnings import warn
from virtual_lab.const import * 

class Experiment:
    def __init__(self,name, exp_const, custom_const = DEFAULT_NUCLEATION, **kwargs):
        self.name = name
        self.title = kwargs.get('title',name)
        self.quantities = {}
        for key,value in exp_const.items():
            setattr(self,key,[[onset,onset+duration] for onset,duration in value])
            self.quantities[key] = getattr(self,key)
        self.dt = kwargs.get("dt",CONSTANTS['dt'])
        self.T = kwargs.get("T",CONSTANTS['T'])
        self.has_custom_constants = False
        if custom_const is not None:
            self._custom_constants = custom_const
            self.set_custom_constants(custom_const)
    
    steps = property(lambda self: int(self.T/self.dt))
    
    def get_custom_constants(self):
        return self._custom_constants
    
    def __call__(self):
        return self.quantities
    
    def __str__(self):
        return self.name

    def set_custom_constants(self,const):
        for key,value in const.items():
            setattr(self,key,value)
        self._custom_constants = const
        # This allows us to run stochastic simulations faster
        self.has_custom_constants = True
    
    def copy(self):
        exp_const = {}
        for name, q in self.quantities.items():
            new_q = []
            for ton,toff in q:
                duration = toff-ton
                new_q.append([ton,duration])
            exp_const[name] = new_q

        return Experiment(self.name,exp_const,nucleation_const=self.get_custom_constants(),T = self.T,dt = self.dt)
    
    def add_event(self,name,intervals):
        setattr(self,name,[[onset,onset+duration] for onset,duration in intervals])
        self.quantities[name] = getattr(self,name)


class Experiments:
    def __init__(self,experiments = None):
        """
        Just a simple container for various experiments
        """
        self.experiments = []
        if experiments is not None:
            self.add(experiments)
    
    def set_custom_constants(self,const):
        for experiment in self.experiments:
            experiment.set_custom_constants(const)

    def add(self, experiments):
        try:
            len(experiments)
        except TypeError:
            experiments = [experiments]
        for experiment in experiments:
            if not isinstance(experiment,Experiment):
                raise TypeError("The experiments should be"
                "instances of the Experiment class, instead I found a"
                f"{type(experiment)} type")
            self.experiments.append(experiment)

    def __iter__(self):
        return iter(self.experiments)
    
    def __len__(self):
        return len(self.experiments)

    def get_experiment(self, _name: str):
        for experiment in self.experiments:
            if experiment.name == _name:
                return experiment
        raise Warning("Experiment wasn't found. Make sure you are using the" 
                    "correct name")

    def __str__(self):
        return [str(experiment) for experiment in self.experiments]

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
                    print(type(intervals[0]))
                    raise NotImplementedError("If an event is concurring with another and you wish to add it, use the format [onset,duration] (no nested lists!)")

BasicExperiments = Experiments()
NO_STIMULI = Experiment("NO_STIMULI",
                {"protein" : [], 
                "xlinkers": [], 
                "stim": [], 
                "LFS": []})
BasicExperiments.add(NO_STIMULI)
W_TET = Experiment("W_TET",
                  {"protein" : [], 
                  "xlinkers": [[300,X_WTET]], 
                  "stim": [[300,WTET]], 
                  "LFS": []}, T = 3*3600)
BasicExperiments.add(W_TET)
S_TET = Experiment("S_TET",
                    {"stim": [[300,STET]], 
                    "LFS": [], 
                    "xlinkers": [[300,X_STET]], 
                    "protein" : [[300+P_ONSET,PROTEIN]] })

BasicExperiments.add(S_TET)
LTP_X2 = Experiment("LTP_X2",
                    {"stim": [[300, STET],[8100, STET]], 
                    "LFS": [], 
                    "xlinkers": [[300, X_STET],[8100, X_STET]], 
                    "protein" : [[300+P_ONSET, PROTEIN],[8100+P_ONSET, PROTEIN]] }, T = 4.5*3600)

BasicExperiments.add(LTP_X2)
MULTI_LTP_BEFORE_PRP = Experiment("MULTI_LTP_BEFORE_PRP",
                                  {"stim": [[300, STET],[300 + STET + 100, STET]], 
                                    "LFS": [], 
                                    "xlinkers": [[300, X_STET],[300 + X_STET + 100, X_STET]], 
                                    "protein" : [[300+P_ONSET, PROTEIN]] })
                                
BasicExperiments.add(MULTI_LTP_BEFORE_PRP)
STC_LTP_SBW = Experiment("STC_LTP_SBW",
                         {"stim": [[900, WTET]], 
                        "LFS": [], 
                        "xlinkers": [[900, X_WTET]], 
                        "protein" : [[0, PROTEIN]]})  

BasicExperiments.add(STC_LTP_SBW)
STC_LTP_WBS = Experiment("STC_LTP_WBS",
                        {"stim": [[300, WTET]],
                        "LFS": [], 
                        "xlinkers": [[300, X_WTET]],
                        "protein" : [[6300, PROTEIN]] }, T = 3*3600)  

BasicExperiments.add(STC_LTP_WBS)
STC_TR5 = Experiment("STC_TR5",
                     {"stim": [[300, STET]],
                        "LFS": [[600, LFS]],
                        "xlinkers": [[300, X_STET], [600, X_LFS]],
                        "protein" : [[300+P_ONSET, PROTEIN]] })  

BasicExperiments.add(STC_TR5)
STC_TR15 = Experiment("STC_TR15",
                      {"stim": [[300, STET]],
                        "LFS": [[1200, LFS]],
                        "xlinkers": [[300, X_STET],[1200, X_LFS]],
                        "protein" : [[300+P_ONSET, PROTEIN]] })  

BasicExperiments.add(STC_TR15)
LTD = Experiment("LTD",
                 {"stim": [],
                        "LFS": [[300, LFS]],
                        "xlinkers": [[300, X_LFS]],
                        "protein" : [[300+P_ONSET, PROTEIN]] })  

BasicExperiments.add(LTD)
STC_LTD_WBS = Experiment("STC_LTD_WBS",
                         {"stim": [],
                        "LFS": [[300, LFS]],
                        "xlinkers": [[300, X_LFS]],
                        "protein" : [[4500, PROTEIN]] },T = 3*3600)  

BasicExperiments.add(STC_LTD_WBS)
STC_LTD_SBW = Experiment("STC_LTD_SBW",
                         {"stim": [],
                        "LFS": [[900, LFS]],
                        "xlinkers": [[900, X_LFS]],
                        "protein" : [[0, PROTEIN]] })  

BasicExperiments.add(STC_LTD_SBW)
XLINKERS = Experiment("XLINKERS",
                      {"stim": [],
                        "LFS": [],
                        "xlinkers": [[300, X_STET]],
                        "protein" : [] })  

BasicExperiments.add(XLINKERS)
BasicExperiments.set_custom_constants(DEFAULT_NUCLEATION)
# Experiments for Metaplasticity
MetaPlasticityExperiments = Experiments()
# Time required to complete one plasticity event
pt = 2*3600
n_LTP = 300
n_LTD = 40
n_LTP_mixed = 100
n_LTD_mixed = 100
LTP_META = Experiment("LTP_META",
                     {"stim" : [[k*pt+10,STET] for k in range(n_LTP)],
                     "LFS" : [],
                     "xlinkers": [[k*pt+10,X_STET] for k in range(n_LTP)],
                     "protein": [[k*pt+P_ONSET,PROTEIN] for k in range(n_LTP) ]
                     }, T = pt*n_LTP, dt = 10)
MetaPlasticityExperiments.add(LTP_META)
LTD_META = Experiment("LTD_META",
                     {"stim" : [],
                     "LFS" : [[k*pt+10,LFS] for k in range(n_LTD)],
                     "xlinkers": [[k*pt+10,X_LFS] for k in range(n_LTD)],
                     "protein": [[k*pt+P_ONSET,PROTEIN] for k in range(n_LTD) ]
                     }, T = pt*n_LTD, dt = 10)
MetaPlasticityExperiments.add(LTD_META)

xlinkers_mixed_metaplaticity = [[k*pt+10,X_STET] for k in range(n_LTP_mixed)]
for k in range(n_LTD_mixed):
    xlinkers_mixed_metaplaticity.append([k*pt+10,X_LFS])
MIX_META = Experiment("MIX_META",
                     {"stim" : [[k*pt+10,STET] for k in range(n_LTP_mixed)],
                     "LFS" : [[k*pt+pt*n_LTP_mixed+10,LFS] for k in range(n_LTD_mixed)],
                     "xlinkers": xlinkers_mixed_metaplaticity,
                     "protein": [[k*pt+P_ONSET,PROTEIN] for k in range(n_LTD_mixed+n_LTP_mixed) ]
                     }, T = pt*(n_LTD_mixed+n_LTP_mixed), dt = 10)
MetaPlasticityExperiments.add(MIX_META)
MetaPlasticityExperiments.set_custom_constants(DEFAULT_NUCLEATION)
# Experiments for Tag resetting
import numpy as np
TagResetExperiments = Experiments([Experiment(f"STC_TR{int(t/60)}",
           {"stim": [[300,STET]],
           "LFS": [[300+t,LFS]],
           "xlinkers": [[300,X_STET],[300+t,X_LFS]],
           "protein": [[300+P_ONSET,PROTEIN]]},T = 2*3600 + t,dt = 5) for t in np.arange(120,2700,120)])






