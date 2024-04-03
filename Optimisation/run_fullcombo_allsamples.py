from helpers import files_from_path, load_cfg_file
import os
from scipy.optimize import minimize
from HLTClass.DiTauJetDataset import DiTauJetDataset
import numpy as np
from Optimisation.helpers_opt import loss
import pandas as pd
import json

deeptau = True

from run_fullcombo import Threshold_optimiser as Th_pnet
from run_fulldeeptau import Threshold_optimiser as Th_deeptau

if __name__ == '__main__':
    config = load_cfg_file()
    RefRun = int(config['RUNINFO']['ref_run'])
    HLT_name = config['HLT']['HLTname']
    L1A_physics = float(config['RUNINFO']['L1A_physics'])
    if not deeptau:
        # with open("results_tri_optimised.json") as fil:
            # d = json.load(fil)

        df = pd.read_pickle("results_ditaujet.pickle")

        df = df[(df['eff'] > 0.679) & (df['eff'] < 0.68) & (df['rate'] > 60 ) & (df['rate'] < 61)]

    # FileNameList_eff = f"/eos/user/j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/{HLT_name}/ZprimeToTauTau_M-4000.root" 
    FileNameList_eff = [
        "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/GluGluHToTauTau_M-125.root",
        "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00.root",
        "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHHto2B2Tau_CV-1_C2V-1_C3-1.root",
        "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHToTauTau_M125.root",
    ]

    if deeptau:
        th = Th_deeptau()
    else:
        th = Th_pnet()

    # print(df["params"])

    Rate_path = os.path.join(config['DATA']['RateDenPath'], f'Run_{RefRun}')
    FileNameList_rate = files_from_path(Rate_path)
    FileNameList_rate = FileNameList_rate[0]
    th.dataset_rate = DiTauJetDataset(FileNameList_rate)

    if not deeptau:
        results = {"params": df["params"], "rate": df["rate"]}
    else:
        results = {"params": "DeepTau"}
        #results = {"params": "DeepTau", "rate": df["rate"]}
    for fil in FileNameList_eff:
        name = fil.split("/")[-1].split(".root")[0]
        results[name] = []
        th.dataset_eff = DiTauJetDataset(fil)
        if not deeptau:
            for params in df["params"]:
                name, print(params)
                results[name].append(
                    th.f(
                        (float(params[1]), float(params[2])), params[0]
                    )[0]
                )
        else:
            results[name].append(th.f()[0])

    results_df = pd.DataFrame(results)

    print(results_df)
