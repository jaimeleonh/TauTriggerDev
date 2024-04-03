from helpers import files_from_path, load_cfg_file
import os
from scipy.optimize import minimize
from HLTClass.DiTauJetDataset import DiTauJetDataset
from HLTClass.dataset import get_L1Taus, get_L1Jets, get_Taus, get_Jets, get_GenTaus, get_GenJets, hGenTau_selection, hGenJet_selection, matching_Gentaus, matching_Genjets, matching_L1Taus_obj, matching_GenObj_l1only
import numpy as np
from Optimisation.helpers_opt import loss
import pandas as pd
import awkward as ak

from HLTClass.DiTauDataset import evt_sel_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1, evt_sel_DiTau, Denominator_Selection_DiTau, L1Tau_IsoTau34er2p1_selection,L1Tau_Tau70er2p1_selection, L1Tau_L2NN_selection_DiTau, Jet_selection_DiTau
from HLTClass.DiTauDataset import get_selL1Taus as get_selL1TausDiTau
from HLTClass.SingleTauDataset import evt_sel_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3, evt_sel_SingleTau, Denominator_Selection_SingleTau, Jet_selection_SingleTau,L1Tau_Tau120er2p1_selection, L1Tau_L2NN_selection_SingleTau
from HLTClass.DiTauJetDataset import evt_sel_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60, evt_sel_DiTauJet, Denominator_Selection_DiTauJet, L1Tau_IsoTau26er2p1_selection, L1Tau_L2NN_selection_DiTauJet, Jet_selection_DiTauJet, get_selL1Jets, L1Jet_Jet55_selection, Jet_selection_DiTauJet_Jets
from HLTClass.DiTauJetDataset import get_selL1Taus as get_selL1TausDiTauJet

par_frozen_SingleTau = [1.0, 0.95, 0.001]
par_frozen_DoubleTau = [0.56, 0.47, 0.001]

class Threshold_optimiser():

    config = load_cfg_file()
    RefRun = int(config['RUNINFO']['ref_run'])
    HLT_name = config['HLT']['HLTname']
    L1A_physics = float(config['RUNINFO']['L1A_physics'])

    def f(self, a, pt=28, apply_doubletaujet=True):
        a = np.append(a, [0])
        events_rate = self.dataset_rate.get_events()
        SingleTau_evt_mask, _ = evt_sel_SingleTau(events_rate, par_frozen_SingleTau, is_gen = False)
        DiTau_evt_mask, _ = evt_sel_DiTau(events_rate, par_frozen_DoubleTau, n_min=2, is_gen = False)
        if apply_doubletaujet:
            DiTauJet_evt_mask, _, _ = evt_sel_DiTauJet(events_rate, a, n_min=2, is_gen = False, pt=pt)
            evt_mask = SingleTau_evt_mask | DiTau_evt_mask | DiTauJet_evt_mask
        else:
            evt_mask = SingleTau_evt_mask | DiTau_evt_mask
        N_den = len(events_rate)
        N_num = len(events_rate[evt_mask])
        rate = (N_num/N_den)*self.L1A_physics

        ############### Efficiency ################
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.dataset_eff.get_events()
        L1Taus = get_L1Taus(events)
        L1Jets = get_L1Jets(events)
        Jets = get_Jets(events)
        GenTaus = get_GenTaus(events)
        GenJets = get_GenJets(events)

        #Select GenTau (same for both)
        GenTau_mask = hGenTau_selection(events)
        GenTaus_Sel = GenTaus[GenTau_mask]

        # Selection of L1 objects and reco Tau objects + matching
        # For SingleTau
        L1Tau_Tau120er2p1L2NN_mask = L1Tau_Tau120er2p1_selection(events) & L1Tau_L2NN_selection_SingleTau(events)
        # SingleTau_mask = Jet_selection_SingleTau(events, par_frozen_SingleTau, apply_PNET_WP = False)
        # at least 1 L1tau/ recoTau should pass
        # SingleTau_evt_mask = (ak.sum(L1Tau_Tau120er2p1L2NN_mask, axis=-1) >= 1) & (ak.sum(SingleTau_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1)
        SingleTau_evt_mask = (ak.sum(L1Tau_Tau120er2p1L2NN_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1)

        L1Taus_SingleTau = L1Taus[L1Tau_Tau120er2p1L2NN_mask]
        # Jets_SingleTau = Jets[SingleTau_mask]
        matching_GenTaus_mask_SingleTau = matching_GenObj_l1only(L1Taus_SingleTau, GenTaus_Sel)
        evt_mask_matching = (ak.sum(matching_GenTaus_mask_SingleTau, axis=-1) >= 1)  # at least 1 Gen Taus should match L1Tau
        SingleTau_evt_mask = SingleTau_evt_mask & evt_mask_matching

        # For DiTau
        L1Tau_IsoTau34er2p1L2NN_mask = L1Tau_IsoTau34er2p1_selection(events) & L1Tau_L2NN_selection_DiTau(events)
        L1Tau_Tau70er2p1L2NN_mask = L1Tau_Tau70er2p1_selection(events) & L1Tau_L2NN_selection_DiTau(events)
        # DiTau_mask = Jet_selection_DiTau(events, par_frozen_DoubleTau, apply_PNET_WP = False)
        # at least n_min L1tau/ recoJet should pass
        # DiTau_evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1L2NN_mask, axis=-1) >= 2) | (ak.sum(L1Tau_Tau70er2p1L2NN_mask, axis=-1) >= 2)) & (ak.sum(DiTau_mask, axis=-1) >= 2) & (ak.sum(GenTau_mask, axis=-1) >= 2)
        DiTau_evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1L2NN_mask, axis=-1) >= 2) | (ak.sum(L1Tau_Tau70er2p1L2NN_mask, axis=-1) >= 2)) & (ak.sum(GenTau_mask, axis=-1) >= 2)

        # matching
        L1Taus_DoubleTau = get_selL1TausDiTau(L1Taus, L1Tau_IsoTau34er2p1L2NN_mask, L1Tau_Tau70er2p1L2NN_mask, n_min_taus = 2)
        # Jets_DoubleTau = Jets[DiTau_mask]
        # matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTau, Jets_DoubleTau, GenTaus_Sel)
        matchingGentaus_mask = matching_GenObj_l1only(L1Taus_DoubleTau, GenTaus_Sel)
        # at least n_min GenTau should match L1Tau/Taus
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= 2)
        DiTau_evt_mask = DiTau_evt_mask & evt_mask_matching

        # For DiTauJet
        L1Tau_IsoTau26er2p1_mask = L1Tau_IsoTau26er2p1_selection(events)
        L1Tau_IsoTau26er2p1L2NN_mask = L1Tau_IsoTau26er2p1_mask & L1Tau_L2NN_selection_DiTauJet(events)
        L1Jet_Jet55_mask = L1Jet_Jet55_selection(events, L1Tau_IsoTau26er2p1_mask)
        # DiTauJet_mask = Jet_selection_DiTauJet(events, a, apply_PNET_WP = False, pt=26)
        # DiTauJet_Jet_mask = Jet_selection_DiTauJet_Jets(events, DiTauJet_mask, usejets=True)

        GenJets_mask = hGenJet_selection(events)
        # at least n_min L1tau/ recoJet and 1 L1jet/jet should pass + L1 trigger
        DiTauJet_evt_mask = (
            # (ak.sum(DiTauJet_mask, axis=-1) >= 2) & (ak.sum(L1Tau_IsoTau26er2p1L2NN_mask, axis=-1) >= 2) & # taus
            (ak.sum(GenTau_mask, axis=-1) >= 2) & (ak.sum(L1Tau_IsoTau26er2p1L2NN_mask, axis=-1) >= 2) & # taus
            # (ak.sum(DiTauJet_Jet_mask, axis=-1) >= 1) & (ak.sum(L1Jet_Jet55_mask, axis=-1) >= 1) # jet
            (ak.sum(GenJets_mask, axis=-1) >= 1) & (ak.sum(L1Jet_Jet55_mask, axis=-1) >= 1) # jet
        )
        # matching
        L1Taus_DoubleTauJet = get_selL1TausDiTauJet(L1Taus, L1Tau_IsoTau26er2p1L2NN_mask, n_min_taus = 2)
        L1Jets_DoubleTauJet = get_selL1Jets(L1Jets, L1Jet_Jet55_mask, n_min_jets = 1)
        # Jets_DoubleTauJet = Jets[DiTauJet_mask]
        # Jets_DoubleTauJet_Jet = Jets[DiTauJet_Jet_mask]

        # matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTauJet, Jets_DoubleTauJet, GenTaus_Sel)
        matchingGentaus_mask = matching_GenObj_l1only(L1Taus_DoubleTauJet, GenTaus_Sel)
        # at least n_min GenTau should match L1Tau/Taus
        DiTauJet_evt_mask = DiTauJet_evt_mask & (ak.sum(matchingGentaus_mask, axis=-1) >= 2)

        # matchingGenjets_mask = matching_Genjets(L1Jets_DoubleTauJet, Jets_DoubleTauJet_Jet, GenJets[GenJets_mask])
        matchingGenjets_mask = matching_GenObj_l1only(L1Jets_DoubleTauJet, GenJets[GenJets_mask])
        # at least 1 GenJet should match L1Jet/Jets
        DiTauJet_evt_mask = DiTauJet_evt_mask & (ak.sum(matchingGenjets_mask, axis=-1) >= 1)

        if True:
            evt_mask = SingleTau_evt_mask | DiTau_evt_mask | DiTauJet_evt_mask
        else:
            evt_mask = SingleTau_evt_mask | DiTau_evt_mask
        N_den = len(events[evt_mask])
        # if True:
            # N_den = len(
                # events[
                    # (ak.sum(L1Tau_Tau120er2p1L2NN_mask, axis=-1) >= 1) |
                    # (ak.sum(L1Tau_IsoTau34er2p1L2NN_mask, axis=-1) >= 2) |
                    # (ak.sum(L1Tau_Tau70er2p1L2NN_mask, axis=-1) >= 2) |
                    # ((ak.sum(L1Tau_IsoTau26er2p1L2NN_mask, axis=-1) >= 2) &
                        # (ak.sum(L1Jet_Jet55_mask, axis=-1) >= 1))
                # ]
            # )
        # else:
            # N_den = len(
                # events[
                    # (ak.sum(L1Tau_Tau120er2p1L2NN_mask, axis=-1) >= 1) |
                    # (ak.sum(L1Tau_IsoTau34er2p1L2NN_mask, axis=-1) >= 2) |
                    # (ak.sum(L1Tau_Tau70er2p1L2NN_mask, axis=-1) >= 2)
                # ]
            # )

        # Selection of L1 objects and reco Tau objects + matching
        # For SingleTau
        SingleTau_mask = Jet_selection_SingleTau(events, par_frozen_SingleTau, apply_PNET_WP = True)
        # at least 1 L1tau/ recoTau should pass
        SingleTau_evt_mask = (ak.sum(L1Tau_Tau120er2p1L2NN_mask, axis=-1) >= 1) & (ak.sum(SingleTau_mask, axis=-1) >= 1) & (ak.sum(GenTau_mask, axis=-1) >= 1)

        Jets_SingleTau = Jets[SingleTau_mask]
        matching_GenTaus_mask_SingleTau = matching_Gentaus(L1Taus_SingleTau, Jets_SingleTau, GenTaus_Sel)
        evt_mask_matching = (ak.sum(matching_GenTaus_mask_SingleTau, axis=-1) >= 1)  # at least 1 Taus should match L1Tau
        SingleTau_evt_mask = SingleTau_evt_mask & evt_mask_matching

        # For DiTau
        DiTau_mask = Jet_selection_DiTau(events, par_frozen_DoubleTau, apply_PNET_WP = True)
        # at least 1 L1tau/ Jet/ GenTau should pass
        DiTau_evt_mask = ((ak.sum(L1Tau_IsoTau34er2p1L2NN_mask, axis=-1) >= 2) | (ak.sum(L1Tau_Tau70er2p1L2NN_mask, axis=-1) >= 2)) & (ak.sum(DiTau_mask, axis=-1) >= 2) & (ak.sum(GenTau_mask, axis=-1) >= 2)

        # matching
        Jets_DoubleTau = Jets[DiTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTau, Jets_DoubleTau, GenTaus_Sel)
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= 2)
        DiTau_evt_mask = DiTau_evt_mask & evt_mask_matching

        if apply_doubletaujet:
            # For DiTauJet
            # Selection of L1/Gen and Jets objects with PNET WP
            DiTauJet_mask = Jet_selection_DiTauJet(events, a, apply_PNET_WP=True, pt=pt)
            DiTauJet_Jet_mask = Jet_selection_DiTauJet_Jets(events, DiTauJet_mask, usejets=True)
            # at least 1 L1tau/ Jet/ GenTau should pass
            DiTauJet_evt_mask = (
                (ak.sum(DiTauJet_mask, axis=-1) >= 2) & (ak.sum(L1Tau_IsoTau26er2p1L2NN_mask, axis=-1) >= 2) & # taus
                (ak.sum(DiTauJet_Jet_mask, axis=-1) >= 1) & (ak.sum(L1Jet_Jet55_mask, axis=-1) >= 1) # jet
            )
            # matching
            # no need to match jets, as they are already included in the denominator
            Jets_DoubleTauJet = Jets[DiTauJet_mask]
            Jets_DoubleTauJet_Jet = Jets[DiTauJet_Jet_mask]
            matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTauJet, Jets_DoubleTauJet, GenTaus_Sel)
            matchingGenjets_mask = matching_Genjets(L1Jets_DoubleTauJet, Jets_DoubleTauJet_Jet, GenJets[GenJets_mask])
            # at least n_min GenTau should match L1Tau/Taus
            evt_mask_matching = evt_mask_matching & (ak.sum(matchingGentaus_mask, axis=-1) >= 2) & (ak.sum(matchingGenjets_mask, axis=-1) >= 1)

            # Or between the 3
            evt_mask = SingleTau_evt_mask | DiTau_evt_mask | DiTauJet_evt_mask
        else:
            evt_mask = SingleTau_evt_mask | DiTau_evt_mask 

        N_num = len(events[evt_mask])

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

    # FileNameList_eff = f"/eos/user/j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/{HLT_name}/ZprimeToTauTau_M-4000.root" 
    FileNameList_eff = [
        "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/GluGluHToTauTau_M-125.root",
        # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/GluGlutoHHto2B2Tau_kl-1p00_kt-1p00_c2-0p00.root",
        # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHHto2B2Tau_CV-1_C2V-1_C3-1.root",
        # "/eos/home-j/jleonhol/www/PNetAtHLT/data/EfficiencyDen/HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60/VBFHToTauTau_M125.root",
    ]
    # FileNameList_eff = f"/afs/cern.ch/work/p/pdebryas/PNetAtHLT/data/EfficiencyDen/{HLT_name}/VBFHToTauTau_M125_ext1.root"
    FileNameList_rate = files_from_path(Rate_path)
    FileNameList_rate = FileNameList_rate[0]
    print(FileNameList_rate)
    #FileNameList_rate = files_from_path(Rate_path)[0] # only one otherwise it's too long (gives good aprosimation for the rate)

    th = Threshold_optimiser()

    th.dataset_eff = DiTauJetDataset(FileNameList_eff)
    th.dataset_rate = DiTauJetDataset(FileNameList_rate)

    min_param_1 = 0.4
    # min_param_1 = 0.59
    min_param_2 = 0.2
    # min_param_2 = 0.59
    max_param = 0.6
    step = 0.01
    effs = []
    rates = []
    params = []

    for i in range(int(round((max_param - min_param_1) / step)) + 1):
        for j in range(int(round((max_param - min_param_2) / step)) + 1):
            param = (min_param_1 + i * step, min_param_2 + j * step)
            if param[1] > param[0]:
                continue
            print(param)
            params.append(param)
            eff, rate = th.f(param)
            effs.append(eff)
            rates.append(rate)

    eff, rate = th.f((0, 0), False)
    print(eff, rate)

    d = {'params': params, 'efficiency': effs, 'rate': rates, 'noditaujet_eff': eff, 'noditaujet_rate': rate}
    df = pd.DataFrame(d)
    res = df.to_json()
    with open("results.json", "w+") as f:
        f.write(res)

