from helpers import files_from_path, load_cfg_file
import os
from scipy.optimize import minimize
from HLTClass.DiTauJetDataset import DiTauJetDataset
import numpy as np
from Optimisation.helpers_opt import loss
import pandas as pd
import json

deeptau = True


def f(a):
        if not deeptau:
            a = np.append(a, [0])

        if not deeptau:
            N_den, N_num = dataset_eff.ComputeEffAlgo_DiTauJetPNet(a)
        else:
            N_den, N_num = dataset_eff.ComputeEffAlgo_HLTDoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60()
        eff_algo = (N_num/N_den)
        
        print(f'Algo Eff: {eff_algo}') 
        print('---------------------------------------------------------------------------------------------------------')
        return eff_algo


if __name__ == '__main__':
    config = load_cfg_file()
    if not deeptau:
        with open("results.json") as fil:
            d = json.load(fil)

        df = pd.DataFrame(d)

        df = df[(df['efficiency'] > 0.84) & (df['efficiency'] < 0.86) & (df['rate'] > 17.8 )& (df['rate'] < 18.2)]

    # FileNameList_eff = f"/eos/user/j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/{HLT_name}/ZprimeToTauTau_M-4000.root" 
    FileNameList_eff = [
        "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/GluGluHToTauTau_M-125.root",
        "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00.root",
        "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHHto2B2Tau_CV-1_C2V-1_C3-1.root",
        "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHToTauTau_M125.root",
    ]

    dataset_eff = DiTauJetDataset(FileNameList_eff)

    if not deeptau:
        results = {"params": df["params"], "rate": df["rate"]}
    else:
        results = {"params": "DeepTau"}
        #results = {"params": "DeepTau", "rate": df["rate"]}
    for fil in FileNameList_eff:
        name = fil.split("/")[-1].split(".root")[0]
        results[name] = []
        dataset_eff = DiTauJetDataset(fil)
        if not deeptau:
            for params in df["params"]:
                results[name].append(
                    f(
                        (float(params[0]), float(params[1]))
                    )
                )
        else:
            results[name].append(f(None))

    results_df = pd.DataFrame(results)

    print(results_df)
