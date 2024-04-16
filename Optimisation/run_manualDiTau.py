from helpers import files_from_path, load_cfg_file
import os
from scipy.optimize import minimize
from HLTClass.DiTauDataset import DiTauDataset
import numpy as np
from Optimisation.helpers_opt import loss

def f(a):
    a = np.append(a, [0])

    N_den, N_num = dataset_rate.get_Nnum_Nden_DiTauPNet(a)
    rate = (N_num/N_den)*L1A_physics

    N_den, N_num = dataset_eff.ComputeEffAlgo_DiTauPNet(a)
    eff_algo = (N_num/N_den)

    print(f'Rate: {rate}')
    print(f'Algo Eff: {eff_algo}') 
    print('---------------------------------------------------------------------------------------------------------')
    return (eff_algo, rate)

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

    min_param = 0.4
    max_param = 0.65
    step = 0.01
    effs = []
    rates = []
    params = []

    for i in range(int(round((max_param - min_param) / step)) + 1):
        for j in range(i + 1):
            param = (min_param + i * step, min_param + j * step)
            # print(param)
            params.append(param)
            eff, rate = f(param)
            effs.append(eff)
            rates.append(rate)

    d = {'params': params, 'efficiency': effs, 'rate': rates}
    df = pd.DataFrame(d)
    res = df.to_json()
    with open("results_ditau.json", "w+") as f:
        f.write(res)

    print(rate_budget)
