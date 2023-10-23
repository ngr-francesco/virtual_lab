# Durations
STET = 120
WTET = 60
PROTEIN = 3600
X_STET = 120
X_WTET = 60
X_LFS = 900
LFS = 900
P_ONSET = 2700 # Bosch 2014 -> The PSD doesn't change in the first 45 mins

CONSTANTS = {
    "dt" : 1,
    "T" : 2*3600,
    # Crosslinker binding rates
    "k_u_0" : 1/3/3600, # Basal transition rates between stable and dynamic pool
    "k_b_0" : 1/3/600, # CaMKII active (unbound) for about 1 min after LTP (Lee et al 2009)
    "k_u_1" : 1/60, # k_u_1 = 1/90
    "k_b_1" : 1/600, # k_b_1 = 1/30
    "t_m_0": 1,
    "t_m_1": 3,
    # Initial PSD volume
    "PSD_0" : 1, 
    # Actin foci
    "AFB" : 0,       # Should nf be multiplied by Vd?
    "nu"  : 0.002,   # Factor at which volume grows with # foci
    # Dynamic actin removal
    "removal" : 2,    # Removal 0: Vd dependent, 1: Vd/PSD dependent; 2: V/PSD dependent
    "DVD"     : 1,     # Should loss in Vd be multiplied by Vd?
    # PSD dynamics
    "tau_PSD" : 360,       # spine growth timescale ~ 6 min; should be 40+ min for PSD95, but 10min for Homer1C (Meyer, 2007)
    "tau_Vd"  : 3600,      # timescale at which volume decays to PSD-related value = Tag-decay ~ 30 min
    "tau_PSD_mol"   : 900,   # PSD molecules are replenished
    "tau_adaptation" : 2200, # The PSD size adapts to its new state (around the time required for L-LTP)
    # Experimental mode: 
    # separate PSD95 and its volume
    # Normally (dynPSD==0), PSD_molecule & volume is acquired when Vs+Vd is enlarged & when protein is synthesized and decays otherwise
    # If dynPSD==1, only V_psd grows when Vd and Vs are unbalanced and otherwise decays to PSD95-molecule-defined volume
    "dynamicPSD" : 0, # use event dynamics for PSD
    # only needed for independent PSD dynamics
    "tau_PSD_str"   : 360,   # PSD structure expansion by dynamic actin (was 3600)
    "tau_PSD_reorg" : 900,  # PSD reorganizes into compact form
    # Nucleation birth-death process dynamics
    "g_mul_LTD" : 1/10, 
    "m_mul_LTD" : 10,
    "g_mul_LTP" : 27, 
    "m_mul_LTP" : 1/15,
    "gamma_0" : 0.02, # For now just making sure nothing explodes
    "mu_0" : 0.5,
    "nf_0" : 0.04,
    "nf_LTP": 15,
    "nf_LTD": 0.,
    ## Duration constants
    # Durations
    "STET" : 120,
    "WTET" : 60,
    "PROTEIN" : 3600,
    "X_STET" : 120,
    "X_WTET" : 60,
    "X_LFS" : 900,
    "X_WLFS": 600,
    "LFS" : 900,
    "WLFS": 600,
    "P_ONSET" : 2700, # Bosch 2014 -> The PSD doesn't change in the first 45 mins
    "E_ONSET" : 420 # 600 # This should be between 2-7 mins
}



import numpy as np
# Lee: first 15 points: 1m = 9.7 mm, then 5 m = 5.88 mm, 100% = 4.06
deltax_lee_0 = np.array([-0.8,1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4,1.4])*1/9.7
deltax_lee_1 = np.array([2.4 for k in range(15)])*5/5.9
deltax_lee = np.append(deltax_lee_0,deltax_lee_1)


EXPERIMENTAL_DATA = {
    "info": """The times are always given in minutes, relative to stimulus, measuring distance between each sample. 
                Volume changes are measured in change relative to initial volume""",
    # Matsuzaki: 20 m = 6.5 mm, 50% = 5.12 mm (reference 100%)
    "Matsuzaki 2004": {"deltax": 20/6.5*np.array([-1.39,1.86,1.76,1.62,1.53,1.66,1.75, 2.4,3.28,3.27, 3.4,3.28, 3.17,3.27]),
                        "y": .5/5.12*np.array([0.2,20.59,8.74,6.45,5.92,6.03,5.00,4.04,3.76,3.83,3.24,4.15,4.36,3.35])+1},
    # Lee: first 15 points: 1m = 9.7 mm, then 5 m = 5.88 mm, 100% = 4.06
    #"sLTP_Lee": {"deltax": deltax_lee,
                 #"y": 1+1/4.1*np.array([4.1,4.95,6.01,7.4,10.3,12.4,13.9,16.0,17.2,18.1,18.5,19.4,19.7,18.5,19.0,17.3,12.6,10.7,9.0,8.8,8.4,8.42,8.0,7.7,8.0,7.7,8.2,7.7,7.6,7.2])-1},
    # Zhou: x is every 15 mins so easy, for y: 25% = 5.4 mm
    #"sLTD_Zhou": { "deltax": np.array([0,15,15,15,15]),
    #                "y": 1-0.25/5.4*np.array([0,3.8,5.8,7.2,6.8])},
    # Kasai Noguchi: 20m = 11 mm, 25% = 8.1 mm
    "Kasai Noguchi 2015": { "deltax": 20/11*np.array([0,5.7,5.5,5.4,5.7,6.0,5.3,11.6]),
                            "y": 1- .25/8.1*np.array([0,4.4,6.3,8.7,8.6,7.9,8.8,8.6])},
    "renorm_var" : "V_tot",
    "time_unit" : "min"
 }

from virtual_lab.math import preprocess_experimental_data
preprocess_experimental_data(EXPERIMENTAL_DATA)


