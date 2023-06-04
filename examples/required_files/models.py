from virtual_lab.model import Model
from virtual_lab.utils import bd_process
import numpy as np

class BaseModel(Model):
    def latex_equations(self):
        """
        Takes the equations defined in the equations_dict and in the diff_equations_dict and translates them from their python
        expressions into latex expressions. This is done by using the sympy library.
        """
        eq = []

        return eq

    def diff_equations_dict(self):
        #super().list_of_equations(variables)
        needed_variables = ["Vd","Vs","Vpsd"]
        for name in needed_variables:
            if not name in self.variables.varnames:
                raise ValueError(f"The variable {name} is needed for this model, but was not given at initialisation")
        return {
            "Vd": self.dVd,
            "Vs": self.dVs,
            "Vpsd": self.dPSD
        }
    def equations_dict(self):
        needed_variables = ["V_eff"]
        for name in needed_variables:
            if not name in self.variables.varnames:
                raise ValueError(f"The variable {name} is needed for this model, but was not given at initialisation")
        return {
            "V_eff": self.V_eff_calc,
            "V_tot": self.V_tot_calc
        }

    def set_initial_values(self,stochastic_simulation):
        self.eql  = self.k_u_0/self.k_b_0 / (1+self.k_u_0/self.k_b_0)
        self.eql2  = self.k_u_1/self.k_b_1 / (1+self.k_u_1/self.k_b_1)
        if stochastic_simulation:
            # This is a bit of a cheat, but I'll check if without this it still works.
            # TODO: check!
            self.nf_0 = np.mean(self.nf)
        self.variables.V_eff = self.nu*self.nf_0*self.tau_Vd + self.variables.Vpsd 
        self.variables.Vd =  self.eql    * self.variables.V_eff
        self.variables.Vs = (1-self.eql) * self.variables.V_eff
        self.Vd_0 = self.variables.Vd 
        self.variables.V_tot = self.variables.Vs + self.variables.Vd
    
    def quantity_dependencies(self):
        needed_quantities = ["k_u_1","k_b_1","nf_LTD","nf_LTP"]
        for q in needed_quantities:
            if not hasattr(self,q):
                raise AttributeError(f"The following quantity is not defined in the model, but it is needed by the experimental procedure {q}\n"
                "Make sure you define all the needed constants before running a simulation")
        dependencies = {
            "k_u": {"crosslink": self.k_u_1},
            "k_b": {"crosslink": self.k_b_1},
            "protein": {"protein": 1},
            "nf": {"stim": self.nf_LTP,
                     "LFS": self.nf_LTD,
                     "cytochalasin": self.nf_LTD},
            "gamma": {"stim": self.g_mul_LTP*self.gamma_0,
                      "LFS": self.g_mul_LTD*self.gamma_0},
            "mu" : {"stim": self.m_mul_LTP*self.mu_0,
                    "LFS": self.m_mul_LTD*self.mu_0},
            "t_m": {"jaspl": self.t_m_1}
        }
        return dependencies
    
    def stochastic_variables_dict(self):
        stochastic_variables = {
            "nf" : (bd_process,["gamma","mu"])
        }
        return stochastic_variables
            
    def dVd(self,t): # Vd with V-dependent removal: Dynamic pool shrinks if V>PST_str, Growth with number of foci 
        Vs = self.variables.Vs
        Vd = self.variables.Vd
        Vpsd = self.variables.Vpsd
        return self.nu*self.nf[t]*(Vd/self.Vd_0)**self.AFB  - (Vd+Vs-Vpsd)/(self.t_m[t]*self.tau_Vd)*(Vd/self.Vd_0)**self.DVD  - self.k_b[t]*Vd + self.k_u[t] *Vs
    
    def dVs(self,t):
        Vs = self.variables.Vs
        Vd = self.variables.Vd
        return self.k_b[t]*Vd - self.k_u[t] *Vs 

    def dPSD(self,t):
        Vs = self.variables.Vs
        Vd = self.variables.Vd
        Vpsd = self.variables.Vpsd
        V_eff = self.variables.V_eff
        PSD = self.protein[t]*(Vd+Vs - V_eff)/(self.tau_PSD_mol)
        return PSD

    def V_eff_calc(self,t,**kwargs):
        PSD = kwargs.get('attractor',self.variables.Vpsd)
        pre = PSD*self.k_u_0
        root = self.k_u_0*(4*self.Vd_0*self.nu*self.nf_0*(self.t_m[t]*self.tau_Vd)*(self.k_u_0 + self.k_b_0)+self.k_u_0*PSD**2)
        root = np.sqrt(root) if root>0 else 0
        den = 2*self.k_u_0
        #print("V_eff calculated", pre+root/den)
        return (pre + root)/den
    
    def V_eff_linear(self,t,**kwargs):
        return self.nu*self.nf[t]*(self.t_m[t]*self.tau_Vd) + self.variables.Vpsd
    
    def V_tot_calc(self,t):
        return self.variables.Vd + self.variables.Vs

class MomentumModel(BaseModel):
    def latex_equations(self):
        eq = [r'$\frac{dV_d}{dt} = \mathrm{b}n_f(t) + \left(\frac{V_{e}-V_s-V_d}{\tau_V}\right)\frac{V_d}{V_{d,eq}} + k_uV_s - k_bV_d$',
              r'$\frac{dV_s}{dt} = k_bV_d - k_uV_s$',
              r'$\frac{dV_e}{dt} = \frac{k_b}{k_u}\frac{V_{tot}-V_eff}{\tau_{V_e}} + \frac{V_{psd}- V_e}{\tau_{decay}}$',
              r'$\frac{dV_{PSD}}{dt} = \phi(t) \frac{V_e-V_{psd}}{\tau_P}$']
        return eq
        
    def diff_equations_dict(self):
        _dict = super().diff_equations_dict()
        needed_variables = ["Ve"]
        for name in needed_variables:
            if not name in self.variables.varnames:
                raise ValueError(f"The variable {name} is needed for this model")
        _dict["Ve"] = self.dVe
        return _dict
    
    def set_initial_values(self,stochastic_model):
        super().set_initial_values(stochastic_model)
        self.variables.Ve = self.variables.Vpsd
    
    
    def dVd(self, t):
        Vs = self.variables.Vs
        Vd = self.variables.Vd
        Vpsd = self.variables.Vpsd
        Ve = self.variables.Ve
        return self.nu*self.nf[t] - (Vd+Vs-Ve)/self.tau_Vd*(Vd/self.Vd_0)  - self.k_b[t] *Vd  + self.k_u[t] *Vs
 
    def dVe(self,t):
        V_tot = self.variables.V_tot
        V_eff = self.variables.V_eff
        Ve = self.variables.Ve
        Vd = self.variables.Vd
        Vs = self.variables.Vs
        Vpsd = self.variables.Vpsd
        return self.k_b[t]/self.k_u[t]*(V_tot - V_eff)/(self.tau_add_Ve) + (Vpsd - Ve)/(self.tau_Ve)
    
    def dPSD(self,t):
        Vs = self.variables.Vs
        Vd = self.variables.Vd
        V_eff = self.variables.V_eff
        Ve = self.variables.Ve
        Vpsd = self.variables.Vpsd
        PSD = self.protein[t]*(Ve - Vpsd)/(self.tau_PSD_mol) # ((np.sign(Ve-Vpsd)-1)/2)*(Vpsd-Ve)/self.tau_Ve
        return PSD
    
    def V_eff_calc(self, t):
        return super().V_eff_calc(t,attractor = self.variables.Ve)

