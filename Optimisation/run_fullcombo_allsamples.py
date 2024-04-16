from helpers import files_from_path, load_cfg_file
import os
from scipy.optimize import minimize
from HLTClass.DiTauJetDataset import DiTauJetDataset
import numpy as np
from Optimisation.helpers_opt import loss
import pandas as pd
import json
import threading

deeptau = False

from run_fullcombo import Threshold_optimiser as Th_pnet
from run_fulldeeptau import Threshold_optimiser as Th_deeptau

from run_manualDiTau_allsamples import EffRateRun as EffRateRunDiTau

class EffRateRun(EffRateRunDiTau):

    eff_min = 0.686
    eff_max = 0.69
    rate_min = 70
    rate_max = 71

    df_name = "results_ditaujet.pickle"
    dataset_class_name = "DiTauJetDataset"
    deeptau_eff_function_name = "NOT_IMPLEMENTED"

    get_effs = lambda self, x, y: x.f((float(y[1]), float(y[2])), y[0])[0]

    HLT_name = "HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60"
    FileNameList_eff = [
        f"/eos/user/j/jleonhol/www/PNetAtHLT/2024_v1/data/EfficiencyDen/{HLT_name}/GluGluHToTauTau_M-125.root",
        f"/eos/user/j/jleonhol/www/PNetAtHLT/2024_v1/data/EfficiencyDen/{HLT_name}/GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00.root",
        # f"/eos/user/j/jleonhol/www/PNetAtHLT/2024_v1/data/EfficiencyDen/{HLT_name}/VBFHHto2B2Tau_CV-1_C2V-1_C3-1.root",
        f"/eos/user/j/jleonhol/www/PNetAtHLT/2024_v1/data/EfficiencyDen/{HLT_name}/VBFHToTauTau_M125.root",
    ]

    def run(self):
        if deeptau:
            th = Th_deeptau()
        else:
            th = Th_pnet()

        if not deeptau:
            # with open("results_tri_optimised.json") as fil:
                # d = json.load(fil)

            df = pd.read_pickle(self.df_name)
            df = df[(df['eff'] > self.eff_min) & (df['eff'] < self.eff_max) & (df['rate'] > self.rate_min) & (df['rate'] < self.rate_max)]

        th.dataset_rate = eval(f"{self.dataset_class_name}(self.FileNameList_rate)")

        if not deeptau:
            results = {"params": df["params"], "rate": df["rate"]}
        else:
            results = {"params": "DeepTau"}

        def get_efficiencies(fil):
            name = fil.split("/")[-1].split(".root")[0]
            th.dataset_eff = eval(f"{self.dataset_class_name}(fil)")
            if not deeptau:
                for params in df["params"]:
                    eff = self.get_effs(th, params)
                    results[name].append(eff)
            else:
                N_den, N_num = eval(f"dataset_eff.{self.deeptau_eff_function_name}()")
                eff = (N_num/N_den)
                results[name].append(eff)

        threads = list()
        for fil in self.FileNameList_eff:
            name = fil.split("/")[-1].split(".root")[0]
            results[name] = []
            get_efficiencies(fil)
            # thread = threading.Thread(target=get_efficiencies, args=(fil,))
            # threads.append(thread)
            # thread.start()

        # for th in threads:
            # th.join()

        results_df = pd.DataFrame(results)
        print(results_df)

if __name__ == '__main__':

    c = EffRateRun()
    c.run()






# if __name__ == '__main__':
    # config = load_cfg_file()
    # RefRun = int(config['RUNINFO']['ref_run'])
    # HLT_name = config['HLT']['HLTname']
    # L1A_physics = float(config['RUNINFO']['L1A_physics'])
    # if not deeptau:
        # # with open("results_tri_optimised.json") as fil:
            # # d = json.load(fil)

        # df = pd.read_pickle("results_ditaujet.pickle")

        # df = df[(df['eff'] > 0.679) & (df['eff'] < 0.68) & (df['rate'] > 60 ) & (df['rate'] < 61)]

    # # FileNameList_eff = f"/eos/user/j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/{HLT_name}/ZprimeToTauTau_M-4000.root" 
    # FileNameList_eff = [
        # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/GluGluHToTauTau_M-125.root",
        # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00.root",
        # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHHto2B2Tau_CV-1_C2V-1_C3-1.root",
        # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHToTauTau_M125.root",
    # ]

    # if deeptau:
        # th = Th_deeptau()
    # else:
        # th = Th_pnet()

    # # print(df["params"])

    # Rate_path = os.path.join(config['DATA']['RateDenPath'], f'Run_{RefRun}')
    # FileNameList_rate = files_from_path(Rate_path)
    # FileNameList_rate = FileNameList_rate[0]
    # th.dataset_rate = DiTauJetDataset(FileNameList_rate)

    # if not deeptau:
        # results = {"params": df["params"], "rate": df["rate"]}
    # else:
        # results = {"params": "DeepTau"}
        # #results = {"params": "DeepTau", "rate": df["rate"]}
    # for fil in FileNameList_eff:
        # name = fil.split("/")[-1].split(".root")[0]
        # results[name] = []
        # th.dataset_eff = DiTauJetDataset(fil)
        # if not deeptau:
            # for params in df["params"]:
                # name, print(params)
                # results[name].append(
                    # th.f(
                        # (float(params[1]), float(params[2])), params[0]
                    # )[0]
                # )
        # else:
            # results[name].append(th.f()[0])

    # results_df = pd.DataFrame(results)

    # print(results_df)
=======
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
>>>>>>> ece4ae826ccd456431518df7ea52bb6d38591823
