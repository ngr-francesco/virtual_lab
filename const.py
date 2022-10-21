# Durations
STET = 60
WTET = 60
PROTEIN = 3600
X_STET = 60
X_WTET = 60
X_ENLARGE = 300 # 300
X_LFS = 1200
LFS = 1200
P_ONSET = 2700 # Bosch 2014 -> The PSD doesn't change in the first 45 mins
E_ONSET = 600 # 600 # This should be between 2-7 mins

CONSTANTS = {
    "dt" : 1,
    "T" : 2*3600,
    # Crosslinker binding rates
    "k_u_0" : 1/2/3600, # Basal transition rates between stable and dynamic pool
    "k_b_0" : 1/2/600, # CaMKII active (unbound) for about 1 min after LTP (Lee et al 2009)
    "k_u_1" : 1/60, # k_u_1 = 1/90
    "k_b_1" : 1/600, # k_b_1 = 1/30
    # Initial PSD volume
    "PSD_0" : 1, 
    # Actin foci
    "AFB" : 0,       # Should nf be multiplied by Vd?
    "nu"  : 0.002,   # Factor at which volume grows with # foci
    "nf_0": 0.03,
    # Dynamic actin removal
    "removal" : 2,    # Removal 0: Vd dependent, 1: Vd/PSD dependent; 2: V/PSD dependent
    "DVD"     : 1,     # Should loss in Vd be multiplied by Vd?
    # PSD dynamics
    "tau_PSD" : 360,       # spine growth timescale ~ 6 min; should be 40+ min for PSD95, but 10min for Homer1C (Meyer, 2007)
    "tau_Vd"  : 3000,      # timescale at which volume decays to PSD-related value = Tag-decay ~ 30 min
    "tau_PSD_mol"   : 900,   # PSD molecules are replenished
    "tau_adaptation" : 2200, # The PSD size adapts to its new state (around the time required for L-LTP)
    # Experimental mode: 
    # separate PSD95 and its volume
    # Normally (dynPSD==0), PSD_molecule & volume is acquired when Vs+Vd is enlarged & when protein is synthesized and decays otherwise
    # If dynPSD==1, only V_psd grows when Vd and Vs are unbalanced and otherwise decays to PSD95-molecule-defined volume
    "dynamicPSD" : 0, # use extra dynamics for PSD
    # only needed for independent PSD dynamics
    "tau_PSD_str"   : 360,   # PSD structure expansion by dynamic actin (was 3600)
    "tau_PSD_reorg" : 900,  # PSD reorganizes into compact form
    # Nucleation birth-death process dynamics
    "g_mul_LTD" : 1/10, 
    "m_mul_LTD" : 10,
    "g_mul_LTP" : 70,  # Bosch 2014
    "m_mul_LTP" : 1/15,
    "gamma_0" : 0.01, # For now just making sure nothing explodes
    "mu_0" : 0.25
}

DEFAULT_NUCLEATION = {
    "nf_0" : 0.03,
    "nf_LTP": 25,
    "nf_LTD": 0,
    "nf_bd_process": False,
    "default" : True
}

