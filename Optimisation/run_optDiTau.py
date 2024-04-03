from helpers import files_from_path, load_cfg_file
import os
from scipy.optimize import minimize
from HLTClass.DiTauDataset import DiTauDataset
import numpy as np
from Optimisation.helpers_opt import loss

def run_optimization(dataset_rate, dataset_eff, rate_budget = 57):
    def f(a):
        a = np.append(a, [0])

        N_den, N_num = dataset_rate.get_Nnum_Nden_DiTauPNet(a)
        rate = (N_num/N_den)*L1A_physics

        N_den, N_num = dataset_eff.ComputeEffAlgo_DiTauPNet(a)
        eff_algo = (N_num/N_den)
        
        print(f'Rate: {rate} vs budget {rate_budget}')
        print(f'Algo Eff: {eff_algo}') 
        print(f'Output score: {- eff_algo + loss(rate, rate_budget)}')
        print('---------------------------------------------------------------------------------------------------------')
        return - eff_algo + loss(rate, rate_budget)

    res = minimize(f, [0.55, 0.47], bounds=((0.5, 0.6), (0.4, 0.5)), method="L-BFGS-B", options={"eps": [0.01, 0.01]})

    return res.x

if __name__ == '__main__':

    config = load_cfg_file()
    RefRun = int(config['RUNINFO']['ref_run'])
    HLT_name = config['HLT']['HLTname']
    L1A_physics = float(config['RUNINFO']['L1A_physics'])

    Rate_path = os.path.join(config['DATA']['RateDenPath'], f'Run_{RefRun}')
    Eff_path = os.path.join(config['DATA']['EffDenPath'], HLT_name)

    FileNameList_eff = f"/eos/user/j/jleonhol/www/PNetAtHLT/2024_v1/data/EfficiencyDen/{HLT_name}/VBFHToTauTau_M125.root"
    FileNameList_rate = files_from_path(Rate_path)[0] # only one otherwise it's too long (gives good aprosimation for the rate)

    dataset_eff = DiTauDataset(FileNameList_eff)
    dataset_rate = DiTauDataset(FileNameList_rate)

    N_den, N_num = dataset_rate.get_Nnum_Nden_HLTDoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1()
    rate_budget = (N_num/N_den)*L1A_physics

    optim_x = run_optimization(dataset_rate, dataset_eff, rate_budget)

    print('Optimisation finished:')
    print("Optimized parameters:", optim_x)
    optim_x = np.append(optim_x, [100, 0])

    N_den, N_num = dataset_rate.get_Nnum_Nden_DiTauPNet(optim_x)
    rate_opt = (N_num/N_den)*L1A_physics

    print("Approx. Rate:", rate_opt)
