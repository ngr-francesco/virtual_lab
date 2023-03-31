
import json
from virtual_lab.experiments import *
from examples.required_files.const import *
with open("experimental_data.json","r") as file:
    exp_data = json.load(file)

# TODO: this looks terrible, I have to implement a better experimental data management system
data_LTP = {}
names = [name for name in exp_data if "Matsuzaki" in name]
names.append("renorm_var")
names.append("time_unit")
for name in names:
    data_LTP[name] = exp_data[name]
data_LTD = {}   
names = [name for name in exp_data if "Kasai" in name]
names.append("renorm_var")
names.append("time_unit")

for name in names:
    data_LTD[name] = exp_data[name]
from virtual_lab.model import Model
def create_experiments(model:Model, **kwargs):
    return_Base = kwargs.get("Basic",True)
    return_STC = kwargs.get("STC",True)
    return_TR = kwargs.get("TR",True)
    return_Meta = kwargs.get("Meta", True)
    return_TR_ext = kwargs.get("TR_ext", True)
    return_Spacing = kwargs.get("Spacing", True)
    return_Multi = kwargs.get("Multi",True)
    returns = {}
    STET = model.STET
    WTET = model.WTET
    PROTEIN = model.PROTEIN
    X_STET = model.X_STET
    X_WTET = model.X_WTET
    X_LFS = model.X_LFS
    WLFS = model.WLFS
    X_WLFS = model.X_WLFS
    LFS = model.LFS
    P_ONSET = model.P_ONSET
    BasicExperiments = Experiments()
    NO_STIMULI = Experiment("NO_STIMULI",
                    {"protein" : [], 
                    "crosslink": [], 
                    "stim": [], 
                    "LFS": []},T = 10*3600)
    BasicExperiments.add(NO_STIMULI)
    ONLY_XLINKERS = Experiment("ONLY_XLINKERS",
                    {"protein" : [], 
                    "crosslink": [[300,120]], 
                    "stim": [], 
                    "LFS": []},T = 10*3600)
    BasicExperiments.add(ONLY_XLINKERS)
    W_TET = Experiment("W_TET",
                    {"protein" : [], 
                    "crosslink": [[300,X_WTET]], 
                    "stim": [[300,WTET]], 
                    "LFS": []}, T = 10*3600)
    BasicExperiments.add(W_TET)
    S_TET = Experiment("S_TET",
                        {"stim": [[300,STET]], 
                        "LFS": [], 
                        "crosslink": [[300,X_STET]], 
                        "protein" : [[300+P_ONSET,PROTEIN]] },
                        exp_data = data_LTP)

    BasicExperiments.add(S_TET)
    LTP_X2 = Experiment("LTP_X2",
                        {"stim": [[300, STET],[8100, STET]], 
                        "LFS": [], 
                        "crosslink": [[300, X_STET],[8100, X_STET]], 
                        "protein" : [[300+P_ONSET, PROTEIN],[8100+P_ONSET, PROTEIN]] }, T = 4.5*3600)

    BasicExperiments.add(LTP_X2)
    LTP_RESET = Experiment("LTP_RESET",
                        {"stim": [[300, STET]],
                            "LFS": [[7800, LFS]],
                            "crosslink": [[300, X_STET],[7800, X_LFS]],
                            "protein" : [[300+P_ONSET, PROTEIN],[7800+P_ONSET,PROTEIN]]},T = 5*3600)   
    BasicExperiments.add(LTP_RESET)
    MULTI_LTP_BEFORE_PRP = Experiment("MULTI_LTP_BEFORE_PRP",
                                    {"stim": [[300, STET],[300 + STET + 100, STET]], 
                                        "LFS": [], 
                                        "crosslink": [[300, X_STET],[300 + X_STET + 100, X_STET]], 
                                        "protein" : [[300+P_ONSET, PROTEIN]] })
                                    
    BasicExperiments.add(MULTI_LTP_BEFORE_PRP)
    STC_LTP_SBW = Experiment("STC_LTP_SBW",
                            {"stim": [[900, WTET]], 
                            "LFS": [], 
                            "crosslink": [[900, X_WTET]], 
                            "protein" : [[0, PROTEIN]]})  

    BasicExperiments.add(STC_LTP_SBW)
    STC_LTP_WBS = Experiment("STC_LTP_WBS",
                            {"stim": [[300, WTET]],
                            "LFS": [], 
                            "crosslink": [[300, X_WTET]],
                            "protein" : [[6300, PROTEIN]] }, T = 3*3600)  

    BasicExperiments.add(STC_LTP_WBS)
    TRBaseExperiments = Experiments()
    STC_TR5 = Experiment("STC_TR5",
                        {"stim": [[300, STET]],
                            "LFS": [[600, LFS]],
                            "crosslink": [[300, X_STET], [600, X_LFS]],
                            "protein" : [[300+P_ONSET, PROTEIN]] })  

    BasicExperiments.add(STC_TR5)
    TRBaseExperiments.add(STC_TR5)
    STC_TR15 = Experiment("STC_TR15",
                        {"stim": [[300, STET]],
                            "LFS": [[1200, LFS]],
                            "crosslink": [[300, X_STET],[1200, X_LFS]],
                            "protein" : [[300+P_ONSET, PROTEIN]] })  

    BasicExperiments.add(STC_TR15)
    STC_TRLTD = Experiment("STC_TRLTD",
                        {"stim": [[1800, WTET]],
                            "LFS": [[300, LFS]],
                            "crosslink": [[300, X_LFS],[1800, X_WTET]],
                            "protein" : [[300+P_ONSET, PROTEIN]] }, title = "Tag Resetting LTD")  

    BasicExperiments.add(STC_TRLTD)
    TRBaseExperiments.add(STC_TR15)
    LTD = Experiment("LTD",
                    {"stim": [],
                            "LFS": [[300, LFS]],
                            "crosslink": [[300, X_LFS]],
                            "protein" : [[300+P_ONSET, PROTEIN]] },
                            exp_data = data_LTD)  

    BasicExperiments.add(LTD)
    W_LFS = Experiment("W_LFS",
                    {"stim": [],
                            "LFS": [[300, WLFS]],
                            "crosslink": [[300, X_WLFS]],
                            "protein" : [] },
                            T = int(15*3600),dt = 5)  

    BasicExperiments.add(W_LFS)
    STC_LTD_WBS = Experiment("STC_LTD_WBS",
                            {"stim": [],
                            "LFS": [[300, WLFS]],
                            "crosslink": [[300, X_WLFS]],
                            "protein" : [[4500, PROTEIN]] },T = 3*3600)  

    BasicExperiments.add(STC_LTD_WBS)
    STC_LTD_SBW = Experiment("STC_LTD_SBW",
                            {"stim": [],
                            "LFS": [[900, WLFS]],
                            "crosslink": [[900, X_WLFS]],
                            "protein" : [[0, PROTEIN]] })  
    BasicExperiments.add(STC_LTD_SBW)
    import numpy as np
    protein_times = [[max(k-PROTEIN,0),k] for k in np.arange(900,7*3600,300)]
    ProteinTimeLTP = Experiments([
        Experiment("Protein availability LTP",
                    {"stim": [[300,WTET]],
                    "LFS": [],
                    "crosslink": [[300, X_WTET]],
                    "protein" : [p] },dt = 5,T = 8*3600) for p in protein_times
        ])
    ProteinTimeLTD = Experiments([
        Experiment("Protein availability LTD",
                    {"stim": [],
                    "LFS": [[300,WLFS]],
                    "crosslink": [[300, X_WTET]],
                    "protein" : [p] },dt = 5,T = 8*3600) for p in protein_times
        ])
    # Basic STC Experiments separated by stimulus
    STCLTPBase= Experiments([exp for exp in BasicExperiments.experiments if "STC_LTP" in exp.name])
    STCLTDBase = Experiments([exp for exp in BasicExperiments.experiments if "STC_LTD" in exp.name])
    STCSBWBase = Experiments([exp for exp in BasicExperiments.experiments if "SBW" in exp.name])
    STCWBSBase = Experiments([exp for exp in BasicExperiments.experiments if "WBS" in exp.name])
    # Experiments for Metaplasticity
    MetaPlasticityExperiments = Experiments()
    # Time required to complete one plasticity event
    pt = int(3.5*3600)
    n_LTP = 300
    n_LTD = 3
    n_LTP_mixed = 100
    n_LTD_mixed = 100
    LTP_META = Experiment("LTP_META",
                        {"stim" : [[k*pt+10,STET] for k in range(n_LTP)],
                        "LFS" : [],
                        "crosslink": [[k*pt+10,X_STET] for k in range(n_LTP)],
                        "protein": [[k*pt+P_ONSET,PROTEIN] for k in range(n_LTP) ]
                        }, T = pt*n_LTP, dt = 10,title = "LTP Metaplasticity")
    MetaPlasticityExperiments.add(LTP_META)
    LTD_META = Experiment("LTD_META",
                        {"stim" : [],
                        "LFS" : [[k*pt+10,LFS] for k in range(n_LTD)],
                        "crosslink": [[k*pt+10,X_LFS] for k in range(n_LTD)],
                        "protein": [[k*pt+P_ONSET,PROTEIN] for k in range(n_LTD) ]
                        }, T = pt*n_LTD, dt = 10,title = "LTD Metaplasticity")
    # MetaPlasticityExperiments.add(LTD_META)

    xlinkers_mixed_metaplaticity = [[k*pt+10,X_STET] for k in range(n_LTP_mixed)]
    for k in range(n_LTD_mixed):
        xlinkers_mixed_metaplaticity.append([k*pt+pt*n_LTP_mixed+10,X_LFS])
    MIX_META = Experiment("MIX_META",
                        {"stim" : [[k*pt+10,STET] for k in range(n_LTP_mixed)],
                        "LFS" : [[k*pt+pt*n_LTP_mixed+10,LFS] for k in range(n_LTD_mixed)],
                        "crosslink": xlinkers_mixed_metaplaticity,
                        "protein": [[k*pt+P_ONSET,PROTEIN] for k in range(n_LTD_mixed+n_LTP_mixed) ]
                        }, T = pt*(n_LTD_mixed+n_LTP_mixed), dt = 10)
    # MetaPlasticityExperiments.add(MIX_META)
    # Experiments for Tag resetting
    TagResetExperiments = Experiments([Experiment(f"STC_TR{int(t/60)}",
            {"stim": [[300,STET]],
            "LFS": [[300+t,LFS]],
            "crosslink": [[300,X_STET],[300+t,X_LFS]],
            "protein": [[300+P_ONSET,PROTEIN]]},T = 3*3600 + t,dt = 5) for t in np.arange(120,3600,120)])

    LTPSpacingExperiments = Experiments([Experiment(f"LTPx2_{int(t/60)}",
            {"stim": [[300,STET],[300+STET+ t,STET]],
            "LFS": [],
            "crosslink": [[300,X_STET],[300+STET+t,X_STET]],
            "protein": [[300+P_ONSET+(PROTEIN+t)*k,PROTEIN] for k in [0,1]] if t/(300+P_ONSET+PROTEIN)>1 else [[300+P_ONSET,PROTEIN]]},
            T = 5*3600 + t,dt = 5) for t in np.arange(120,7200,120)])
    LTPSpacingExperiments.add(S_TET,idx = 0)
    MultiLTPExperiments = Experiments([
        Experiment(f"MULTI-{n_stim}",
            {"stim": [[300+k*180,STET] for k in range(n_stim)],
            "LFS": [],
            "crosslink": [[300+k*180,X_STET] for k in range(n_stim)],
            "protein": [[300+P_ONSET,PROTEIN]]},T = 3*3600,dt = 5) for n_stim in range(0,11)])
    

    # Returns!
    if return_Base:
        returns["Basic"] = BasicExperiments
    if return_STC:
        returns["STCLTP"] = STCLTPBase
        returns["STCLTD"] = STCLTDBase
        returns["STCSBW"] = STCSBWBase
        returns["STCWBS"] = STCWBSBase
        returns["ProteinTimeLTP"] = ProteinTimeLTP
        returns["ProteinTimeLTD"] = ProteinTimeLTD
    if return_TR:
        returns["TR"] = TRBaseExperiments
    if return_Meta:
        returns["Metaplasticity"] = MetaPlasticityExperiments
    if return_TR_ext:
        returns["TR_extended"] = TagResetExperiments
    if return_Spacing:
        returns["Spacing"] = LTPSpacingExperiments
    if return_Multi:
        returns["Multi_LTP"] = MultiLTPExperiments
    return returns