import awkward as ak
import numpy as np
import uproot
import math
import numba as nb
from HLTClass.dataset import Dataset
from HLTClass.dataset import (
    get_L1Taus, get_L1Jets, get_Taus, get_Jets, get_GenTaus, get_GenJets, hGenTau_selection, hGenJet_selection,
    matching_Gentaus, matching_Genjets, matching_L1Taus_obj, matching_L1Jets_obj, compute_PNet_charge_prob
)

# ------------------------------ functions for DiTau with PNet -----------------------------------------------------------------------------
def compute_PNet_WP_DiTauJet(tau_pt, par):
    # return PNet WP for DiTauJet (to optimize)
    t1 = par[0]
    t2 = par[1]
    t3 = 0.001
    t4 = 0
    x1 = 30
    x2 = 100
    x3 = 500
    x4 = 1000
    PNet_WP = tau_pt*0.
    ones = tau_pt/tau_pt
    PNet_WP = ak.where((tau_pt <= ones*x1) == False, PNet_WP, ones*t1)
    PNet_WP = ak.where((tau_pt >= ones*x4) == False, PNet_WP, ones*t4)
    PNet_WP = ak.where(((tau_pt < ones*x2) & (tau_pt > ones*x1)) == False, PNet_WP, (t2 - t1) / (x2 - x1) * (tau_pt - ones*x1) + ones*t1)
    PNet_WP = ak.where(((tau_pt >= ones*x2) & (tau_pt < ones*x3))== False, PNet_WP, (t3 - t2) / (x3 - x2) * (tau_pt - ones*x2) + ones*t2)
    PNet_WP = ak.where(((tau_pt >= ones*x3) & (tau_pt < ones*x4))== False, PNet_WP, (t4 - t3) / (x4 - x3) * (tau_pt - ones*x3) + ones*t3)
    return PNet_WP

def Jet_selection_DiTauJet(events, par, apply_PNET_WP = True):
    # return mask for Jet (Taus) passing selection for DiTauJet path
    Jet_pt_corr = events['Jet_pt'].compute()*events['Jet_PNet_ptcorr'].compute()
    Jets_mask = (events['Jet_pt'].compute() >= 30) & (np.abs(events['Jet_eta'].compute()) <= 2.3) & (Jet_pt_corr >= 30)
    if apply_PNET_WP:
        probTauP = events['Jet_PNet_probtauhp'].compute()
        probTauM = events['Jet_PNet_probtauhm'].compute()
        Jets_mask = Jets_mask & ((probTauP + probTauM) >= compute_PNet_WP_DiTauJet(Jet_pt_corr, par)) & (compute_PNet_charge_prob(probTauP, probTauM) >= 0)
    return Jets_mask

@nb.jit(nopython=True)
def phi_mpi_pi(x: float) -> float: 
    while (x >= 3.14159):
        x -= (2 * 3.14159)
    while (x < -3.14159):
        x += (2 * 3.14159)
    return x

@nb.jit(nopython=True)
def deltaR(eta1: float, phi1: float, eta2: float, phi2: float) -> float:
    deta = eta1 - eta2
    dphi = phi_mpi_pi(phi1 - phi2)
    return float(math.sqrt(deta * deta + dphi * dphi))

@nb.jit(nopython=True)
def apply_ovrm(builder, tau_eta, tau_phi, jet_pt, jet_eta, jet_phi, jet_pt_th):
    for iev in range(len(tau_eta)):
        builder.begin_list()
        for j_pt, j_eta, j_phi in zip(jet_pt[iev], jet_eta[iev], jet_phi[iev]):
            if j_pt < jet_pt_th:
                builder.append(False)
                continue
            num_matches = 0
            for t_eta, t_phi in zip(tau_eta[iev], tau_phi[iev]):
                if t_eta == None:
                    continue
                
                dR = deltaR(j_eta, j_phi, t_eta, t_phi)
                if dR > 0.5:
                    num_matches += 1
            builder.append(num_matches >= 2)
        builder.end_list()
    return builder

def Jet_selection_DiTauJet_Jets(events, DiTauJet_mask) -> ak.Array:
    # return mask for Jet passing selection for DiTauJet path

    tau_eta = ak.drop_none(ak.mask(events['Jet_eta'].compute(), DiTauJet_mask))
    tau_phi = ak.drop_none(ak.mask(events['Jet_phi'].compute(), DiTauJet_mask))

    jet_pt = events['Jet_pt'].compute()
    jet_eta = events['Jet_eta'].compute()
    jet_phi = events['Jet_phi'].compute()

    return apply_ovrm(ak.ArrayBuilder(), tau_eta, tau_phi, jet_pt, jet_eta, jet_phi, 60.).snapshot()

def evt_sel_DiTauJet(events, par, n_min=2, is_gen = False):
    # Selection of event passing condition of DiTauJet with PNet HLT path + mask of objects passing those conditions

    L1Tau_IsoTau26er2p1_mask = L1Tau_IsoTau26er2p1_selection(events)
    L1Tau_IsoTau26er2p1L2NN_mask = L1Tau_IsoTau26er2p1_mask & L1Tau_L2NN_selection_DiTauJet(events)
    L1Jet_Jet55_mask = L1Jet_Jet55_selection(events, L1Tau_IsoTau26er2p1_mask)
    # L1Tau_Tau70er2p1L2NN_mask = L1Tau_Tau70er2p1_selection(events) & L1Tau_L2NN_selection_DiTau(events)
    DiTauJet_mask = Jet_selection_DiTauJet(events, par, apply_PNET_WP = True)
    DiTauJet_Jet_mask = Jet_selection_DiTauJet_Jets(events, DiTauJet_mask)
    # at least n_min L1tau/ recoJet and 1 L1jet / recoJet should pass
    # applying also the full L1 seed selection to account for the Overlap Removal
    DiTauJet_evt_mask = (
        (ak.sum(DiTauJet_mask, axis=-1) >= n_min) & (ak.sum(L1Tau_IsoTau26er2p1L2NN_mask, axis=-1) >= n_min) & # taus
        (ak.sum(DiTauJet_Jet_mask, axis=-1) >= 1) & (ak.sum(L1Jet_Jet55_mask, axis=-1) >= 1) # jet
    ) 
    
    if is_gen:
        # if MC data, at least n_min GenTau should also pass
        GenTau_mask = hGenTau_selection(events)
        GenJet_mask = hGenJet_selection(events)
        DiTauJet_evt_mask = DiTauJet_evt_mask & (ak.sum(GenTau_mask, axis=-1) >= n_min)
        DiTauJet_evt_mask = DiTauJet_evt_mask & (ak.sum(GenJet_mask, axis=-1) >= 1)

    # matching
    L1Taus = get_L1Taus(events)
    L1Jets = get_L1Jets(events)
    Jets = get_Jets(events)
    L1Taus_DoubleTauJet = get_selL1Taus(L1Taus, L1Tau_IsoTau26er2p1L2NN_mask, n_min_taus = n_min)
    L1Jets_DoubleTauJet = get_selL1Jets(L1Jets, L1Jet_Jet55_mask, n_min_jets = 1)
    Jets_DoubleTauJet = Jets[DiTauJet_mask]
    Jets_DoubleTauJet_Jet = Jets[DiTauJet_Jet_mask]

    if is_gen:
        GenTaus = get_GenTaus(events)
        GenTaus_DoubleTauJet  = GenTaus[GenTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTauJet, Jets_DoubleTauJet, GenTaus_DoubleTauJet)

        GenJets = get_GenJets(events)
        GenJets_DoubleTauJet  = GenJets[GenJet_mask]
        matchingGenjets_mask = matching_Genjets(L1Jets_DoubleTauJet, Jets_DoubleTauJet_Jet, GenJets_DoubleTauJet)

        # at least n_min GenTau should match L1Tau/Taus and 1 GenJet L1Jet/Jets
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= n_min) & (ak.sum(matchingGenjets_mask, axis=-1) >= 1)
    else:
        matchingJets_mask = matching_L1Taus_obj(L1Taus_DoubleTauJet, Jets_DoubleTauJet)
        # at least n_min Tau should match L1Tau
        evt_mask_matching = (ak.sum(matchingJets_mask, axis=-1) >= n_min)

        matchingJets_Jet_mask = matching_L1Jets_obj(L1Jets_DoubleTauJet, Jets_DoubleTauJet_Jet)
        # at least 1 Jet should match L1Jet
        evt_mask_matching = evt_mask_matching & (ak.sum(matchingJets_Jet_mask, axis=-1) >= 1)

    DiTauJet_evt_mask = DiTauJet_evt_mask & evt_mask_matching
    if is_gen: 
        return DiTauJet_evt_mask, matchingGentaus_mask, matchingGenjets_mask
    else:
        return DiTauJet_evt_mask, matchingJets_mask, matchingJets_Jet_mask
    
# ------------------------------ functions for DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60 ---------------------------------------------------
def compute_DeepTau_WP_DiTau(tau_pt):
    # return DeepTau WP for DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60
    t1 = 0.649
    t2 = 0.441
    t3 = 0.05
    x1 = 35
    x2 = 100
    x3 = 300
    Tau_WP = tau_pt*0.
    ones = tau_pt/tau_pt
    Tau_WP = ak.where((tau_pt <= ones*x1) == False, Tau_WP, ones*t1)
    Tau_WP = ak.where((tau_pt >= ones*x3) == False, Tau_WP, ones*t3)
    Tau_WP = ak.where(((tau_pt < ones*x2) & (tau_pt > ones*x1)) == False, Tau_WP, (t2 - t1) / (x2 - x1) * (tau_pt - ones*x1) + ones*t1)
    Tau_WP = ak.where(((tau_pt >= ones*x2) & (tau_pt < ones*x3))== False, Tau_WP, (t3 - t2) / (x3 - x2) * (tau_pt - ones*x2) + ones*t2)
    return Tau_WP

def Tau_selection_DiTauJet(events, apply_DeepTau_WP = True):
    # return mask for Tau passing selection for DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60
    tau_pt = events['Tau_pt'].compute()
    Tau_mask = (tau_pt >= 30) & (np.abs(events['Tau_eta'].compute()) <= 2.1)
    if apply_DeepTau_WP:
        Tau_mask = Tau_mask & (events['Tau_deepTauVSjet'].compute() >= compute_DeepTau_WP_DiTau(tau_pt))
    return Tau_mask

def Jet_selection_DiTauJet(events):
    # return mask for Jet passing selection for DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60
    jet_pt = events['Jet_pt'].compute()
    Jet_mask = (jet_pt >= 60)
    return Jet_mask

def evt_sel_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60(events, n_min = 2, is_gen = False):
    # Selection of event passing condition of DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60 + mask of objects passing those conditions

    L1Tau_IsoTau26er2p1_mask = L1Tau_IsoTau26er2p1_selection(events)
    L1Tau_IsoTau26er2p1L2NN_mask = L1Tau_IsoTau26er2p1_mask & L1Tau_L2NN_selection_DiTauJet(events)
    L1Jet_Jet55_mask = L1Jet_Jet55_selection(events, L1Tau_IsoTau26er2p1_mask)
    # L1Tau_Tau70er2p1L2NN_mask = L1Tau_Tau70er2p1_selection(events) & L1Tau_L2NN_selection_DiTau(events)
    DiTauJet_mask = Tau_selection_DiTauJet(events)
    DiTauJet_Jet_mask = Jet_selection_DiTauJet_Jets(events, DiTauJet_mask)
    # at least n_min L1tau/ recoJet and 1 L1jet / recoJet should pass
    # applying also the full L1 seed selection to account for the Overlap Removal
    DiTauJet_evt_mask = (
        (ak.sum(DiTauJet_mask, axis=-1) >= n_min) & (ak.sum(L1Tau_IsoTau26er2p1L2NN_mask, axis=-1) >= n_min) & # taus
        (ak.sum(DiTauJet_Jet_mask, axis=-1) >= 1) & (ak.sum(L1Jet_Jet55_mask, axis=-1) >= 1) # jet
    )

    # print(events['L1Jet_pt'].compute())
    # print(ak.mask(events['L1Jet_pt'].compute(), L1Jet_Jet55_mask))

    # print("Event:", events['event'].compute()[8898])

    # HLT_mask = HLT_selection(events)
    # events = events['event'].compute()
    # for iev, (ev1, ev2) in enumerate(zip(DiTauJet_evt_mask, HLT_mask)):
        # if not ev1 and ev2:
            # print(iev, events[iev])
    # import sys
    # sys.exit()

    if is_gen:
        # if MC data, at least n_min GenTau should also pass
        GenTau_mask = hGenTau_selection(events)
        GenJet_mask = hGenJet_selection(events)
        DiTauJet_evt_mask = DiTauJet_evt_mask & (ak.sum(GenTau_mask, axis=-1) >= n_min)
        DiTauJet_evt_mask = DiTauJet_evt_mask & (ak.sum(GenJet_mask, axis=-1) >= 1)

    # matching
    L1Taus = get_L1Taus(events)
    L1Jets = get_L1Jets(events)
    Taus = get_Taus(events)
    Jets = get_Jets(events)
    L1Taus_DoubleTauJet = get_selL1Taus(L1Taus, L1Tau_IsoTau26er2p1L2NN_mask, n_min_taus = n_min)
    L1Jets_DoubleTauJet = get_selL1Jets(L1Jets, L1Jet_Jet55_mask, n_min_jets = 1)
    Taus_DoubleTauJet = Taus[DiTauJet_mask]
    Jets_DoubleTauJet_Jet = Jets[DiTauJet_Jet_mask]

    if is_gen:
        GenTaus = get_GenTaus(events)
        GenTaus_DoubleTauJet  = GenTaus[GenTau_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTauJet, Taus_DoubleTauJet, GenTaus_DoubleTauJet)

        GenJets = get_GenJets(events)
        GenJets_DoubleTauJet  = GenJets[GenJet_mask]
        matchingGenjets_mask = matching_Genjets(L1Jets_DoubleTauJet, Jets_DoubleTauJet_Jet, GenJets_DoubleTauJet)

        # at least n_min GenTau should match L1Tau/Taus and 1 GenJet L1Jet/Jets
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= n_min) & (ak.sum(matchingGenjets_mask, axis=-1) >= 1)
    else:
        matchingTaus_mask = matching_L1Taus_obj(L1Taus_DoubleTauJet, Taus_DoubleTauJet)
        # at least n_min Tau should match L1Tau
        evt_mask_matching = (ak.sum(matchingTaus_mask, axis=-1) >= n_min)

        matchingJets_Jet_mask = matching_L1Jets_obj(L1Jets_DoubleTauJet, Jets_DoubleTauJet_Jet)
        # at least 1 Jet should match L1Jet
        evt_mask_matching = evt_mask_matching & (ak.sum(matchingJets_Jet_mask, axis=-1) >= 1)

    DiTauJet_evt_mask = DiTauJet_evt_mask & evt_mask_matching
    if is_gen: 
        return DiTauJet_evt_mask, matchingGentaus_mask, matchingGenjets_mask
    else:
        return DiTauJet_evt_mask, matchingTaus_mask, matchingJets_Jet_mask
# ------------------------------ Common functions for DitauJet path ---------------------------------------------------------------
def L1Tau_IsoTau26er2p1_selection(events):
    # return mask for L1tau passing IsoTau26er2p1 selection
    L1_IsoTau26er2p1_mask = (events['L1Tau_pt'].compute() >= 26) & (events['L1Tau_eta'].compute() <= 2.131) & (events['L1Tau_eta'].compute() >= -2.131) & (events['L1Tau_hwIso'].compute() > 0 )
    return L1_IsoTau26er2p1_mask

def L1Jet_Jet55_selection(events, DiTauJet_mask) -> ak.Array:
    
    tau_eta = ak.drop_none(ak.mask(events['L1Tau_eta'].compute(), DiTauJet_mask))
    tau_phi = ak.drop_none(ak.mask(events['L1Tau_phi'].compute(), DiTauJet_mask))

    jet_pt = events['L1Jet_pt'].compute()
    jet_eta = events['L1Jet_eta'].compute()
    jet_phi = events['L1Jet_phi'].compute()

    return apply_ovrm(ak.ArrayBuilder(), tau_eta, tau_phi, jet_pt, jet_eta, jet_phi, 55.).snapshot()

def L1_DoubleIsoTau26er2p1_Jet55_RmOvlp_dR0p5_selection(events):
    # return mask for L1tau passing IsoTau26er2p1 selection
    #L1_IsoTau34er2p1_mask = (events['L1Tau_hwPt'].compute() >= 0x44) & (events['L1Tau_hwEta'].compute() <= 0x30) & (events['L1Tau_hwEta'].compute() >= -49) & (events['L1Tau_hwIso'].compute() > 0 )
    # L1_IsoTau26er2p1_mask = (events['L1Tau_pt'].compute() >= 26) & (events['L1Tau_eta'].compute() <= 2.131) & (events['L1Tau_eta'].compute() >= -2.131) & (events['L1Tau_hwIso'].compute() > 0 )
    L1_DoubleIsoTau26er2p1_Jet55_RmOvlp_dR0p5_mask = (events['L1_DoubleIsoTau26er2p1_Jet55_RmOvlp_dR0p5'].compute() > 0)
    return L1_DoubleIsoTau26er2p1_Jet55_RmOvlp_dR0p5_mask

def HLT_selection(events):
    # return mask for L1tau passing IsoTau26er2p1 selection
    #L1_IsoTau34er2p1_mask = (events['L1Tau_hwPt'].compute() >= 0x44) & (events['L1Tau_hwEta'].compute() <= 0x30) & (events['L1Tau_hwEta'].compute() >= -49) & (events['L1Tau_hwIso'].compute() > 0 )
    # L1_IsoTau26er2p1_mask = (events['L1Tau_pt'].compute() >= 26) & (events['L1Tau_eta'].compute() <= 2.131) & (events['L1Tau_eta'].compute() >= -2.131) & (events['L1Tau_hwIso'].compute() > 0 )
    HLT_mask = (events['HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60'].compute() > 0)
    return HLT_mask

def L1Tau_L2NN_selection_DiTauJet(events):
    # return mask for L1tau passing L2NN selection for DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1
    L1_L2NN_mask = ((events['L1Tau_l2Tag'].compute() > 0.386) | (events['L1Tau_pt'].compute() >= 250))
    return L1_L2NN_mask

def Denominator_Selection_DiTauJet(GenLepton):
    # return mask for event passing minimal GenTau requirements for diTauJet HLT (2 hadronic Taus with min vis. pt and eta)
    mask = (GenLepton['kind'] == 5)
    ev_mask = ak.sum(mask, axis=-1) >= 2  # at least 2 Gen taus should pass this requirements
    return ev_mask

def Denominator_Selection_DiTauJet_Jet(events):
    # return mask for event passing minimal GenJet requirements for diTauJet HLT (1 jet)
    mask = (events["GenJet_pt"].compute() > 0)
    ev_mask = ak.sum(mask, axis=-1) >= 1  # at least 1 Gen jet should pass this requirements
    return ev_mask

def get_selL1Taus(L1Taus, L1Tau_IsoTau26er2p1_mask, n_min_taus = 2):
    # return L1tau that pass DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60 selection (L1Tau_IsoTau26er2p1)
    IsoTau26er2p1 = (ak.sum(L1Tau_IsoTau26er2p1_mask, axis=-1) >= n_min_taus)
    L1Taus_Sel = L1Taus
    L1Taus_Sel = ak.where(IsoTau26er2p1 == False, L1Taus_Sel, L1Taus[L1Tau_IsoTau26er2p1_mask])
    return L1Taus_Sel

def get_selL1Jets(L1Jets, L1Jet_Jet55_mask, n_min_jets = 1):
    # return L1jet that pass DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60 selection (L1Jet_Jet55)
    Jet55 = (ak.sum(L1Jet_Jet55_mask, axis=-1) >= n_min_jets)
    L1Jets_Sel = L1Jets
    L1Jets_Sel = ak.where(Jet55 == False, L1Jets_Sel, L1Jets[L1Jet_Jet55_mask])
    return L1Jets_Sel

class DiTauJetDataset(Dataset):
    def __init__(self, fileName):
        Dataset.__init__(self, fileName)

    # ------------------------------ functions to Compute Rate ---------------------------------------------------------------------

    def get_Nnum_Nden_HLTDoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60(self):
        print(f'Computing rate for DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60:')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in denominator: {len(events)}")

        DiTauJet_evt_mask, _, _ = evt_sel_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60(events, n_min = 2, is_gen = False)
        N_num = len(events[DiTauJet_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def get_Nnum_Nden_DiTauJetPNet(self, par):
        print(f'Computing Rate for DiTauJet path with param: {par}')
        #load all events in the file that belong to the run and lumiSections_range, save the number of events in Denominator
        events = self.get_events()
        N_den = len(events)
        print(f"Number of events in denominator: {len(events)}")

        DiTauJet_evt_mask, _, _ = evt_sel_DiTauJet(events, par, n_min=2, is_gen = False)
        N_num = len(events[DiTauJet_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def save_Event_Nden_eff_DiTauJet(self, tmp_file):
        #Save only needed informations for events passing minimal Gen requirements for diTau HLT (passing denominator selection for efficiency)
        events = self.get_events()
        print(f"Number of events in the file: {len(events)}")
        GenLepton = self.get_GenLepton(events)
        evt_mask = Denominator_Selection_DiTauJet(GenLepton) & Denominator_Selection_DiTauJet_Jet(events)
        print(f"Number of events with at least 2 hadronic Tau and 1 jet: {ak.sum(evt_mask)}")
        self.Save_Event_Nden_Eff(events, GenLepton, evt_mask, tmp_file)
        return

    def save_info(self, events_Den, events_Num, Tau_Den, Tau_Num, Jet_Den, Jet_Num, out_file):
        # saving infos
        lst_evt_Den = {}
        lst_evt_Den['nPFPrimaryVertex'] = np.array(events_Den['nPFPrimaryVertex'].compute())
        lst_evt_Den['nPFSecondaryVertex'] = np.array(events_Den['nPFSecondaryVertex'].compute())

        lst_evt_Num = {}
        lst_evt_Num['nPFPrimaryVertex'] = np.array(events_Num['nPFPrimaryVertex'].compute())
        lst_evt_Num['nPFSecondaryVertex'] = np.array(events_Num['nPFSecondaryVertex'].compute())

        lst_Den = {}
        lst_Den['Tau_pt'] = Tau_Den.pt
        lst_Den['Tau_eta'] = Tau_Den.eta
        lst_Den['Tau_phi'] = Tau_Den.phi
        lst_Den['Tau_nChargedHad'] = Tau_Den.nChargedHad
        lst_Den['Tau_nNeutralHad'] = Tau_Den.nNeutralHad
        lst_Den['Tau_DecayMode'] = Tau_Den.DecayMode
        lst_Den['Tau_charge'] = Tau_Den.charge
        lst_Den['Jet_pt'] = Jet_Den.pt
        lst_Den['Jet_eta'] = Jet_Den.eta
        lst_Den['Jet_phi'] = Jet_Den.phi

        lst_Num = {}
        lst_Num['Tau_pt'] = Tau_Num.pt
        lst_Num['Tau_eta'] = Tau_Num.eta
        lst_Num['Tau_phi'] = Tau_Num.phi
        lst_Num['Tau_nChargedHad'] = Tau_Num.nChargedHad
        lst_Num['Tau_nNeutralHad'] = Tau_Num.nNeutralHad
        lst_Num['Tau_DecayMode'] = Tau_Num.DecayMode
        lst_Num['Tau_charge'] = Tau_Num.charge
        lst_Num['Jet_pt'] = Jet_Num.pt
        lst_Num['Jet_eta'] = Jet_Num.eta
        lst_Num['Jet_phi'] = Jet_Num.phi

        with uproot.create(out_file, compression=uproot.ZLIB(4)) as file:
            file["eventsDen"] = lst_evt_Den
            file["TausDen"] = lst_Den
            file["eventsNum"] = lst_evt_Num
            file["TausNum"] = lst_Num
        return 

    def produceRoot_HLTDoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60(self, out_file):
        #load all events that pass denominator Selection
        events = self.get_events()
        # To compute efficiency, we save in denominator GenTau which pass minimal selection
        GenTau_mask = hGenTau_selection(events)
        GenTaus = get_GenTaus(events)
        Tau_Den = GenTaus[GenTau_mask]

        mask_den_selection = ak.num(Tau_Den['pt']) >=2
        Tau_Den = Tau_Den[mask_den_selection]
        events = events[mask_den_selection]

        print(f"Number of GenTaus passing denominator selection: {len(ak.flatten(Tau_Den))}")

        GenJet_mask = hGenJet_selection(events)
        GenJets = get_GenJets(events)
        Jet_Den = GenJets[GenJet_mask]

        mask_den_selection = ak.num(Jet_Den['pt']) >=1
        Jet_Den = Jet_Den[mask_den_selection]
        events = events[mask_den_selection]

        print(f"Number of GenJets passing denominator selection: {len(ak.flatten(Jet_Den))}")

        DiTauJet_evt_mask, matchingGentaus_mask, matchingGenjets_mask = evt_sel_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60(events, n_min = 1, is_gen = True)
        Tau_Num = (Tau_Den[matchingGentaus_mask])[DiTauJet_evt_mask]
        print(f"Number of GenTaus passing numerator selection: {len(ak.flatten(Tau_Num))}")

        Jet_Num = (Jet_Den[matchingGenjets_mask])[DiTauJet_evt_mask]
        print(f"Number of GenJets passing numerator selection: {len(ak.flatten(Jet_Num))}")
        events_Num = events[DiTauJet_evt_mask]

        self.save_info(events, events_Num, Tau_Den, Tau_Num, Jet_Den, Jet_Num, out_file)
        return

    def produceRoot_DiTauPNet(self, out_file, par):
        #load all events that pass denominator Selection
        events = self.get_events()

        # To compute efficiency, we save in denominator GenTau which pass minimal selection
        GenTau_mask = hGenTau_selection(events)
        GenTaus = get_GenTaus(events)
        Tau_Den = GenTaus[GenTau_mask]

        mask_den_selection = ak.num(Tau_Den['pt']) >=2
        Tau_Den = Tau_Den[mask_den_selection]
        events = events[mask_den_selection]

        print(f"Number of GenTaus passing denominator selection: {len(ak.flatten(Tau_Den))}")

        GenJet_mask = hGenJet_selection(events)
        GenJets = get_GenJets(events)
        Jet_Den = GenJets[GenJet_mask]

        mask_den_selection = ak.num(Jet_Den['pt']) >=1
        Jet_Den = Jet_Den[mask_den_selection]
        events = events[mask_den_selection]

        print(f"Number of GenJets passing denominator selection: {len(ak.flatten(Jet_Den))}")

        DiTauJet_evt_mask, matchingGentaus_mask, matchingGenjets_mask = evt_sel_DiTauJet(events, par, n_min=1, is_gen = True)

        Tau_Num = (Tau_Den[matchingGentaus_mask])[DiTauJet_evt_mask]
        print(f"Number of GenTaus passing numerator selection: {len(ak.flatten(Tau_Num))}")
        events_Num = events[DiTauJet_evt_mask]

        Jet_Num = (Jet_den[matchingGenjets_mask])[DiTauJet_evt_mask]
        print(f"Number of GenJets passing numerator selection: {len(ak.flatten(Jet_Num))}")
        events_Num = events[DiTauJet_evt_mask]

        self.save_info(events, events_Num, Tau_Den, Tau_Num, Jet_Den, Jet_Num, out_file)

        return

    # ------------------------------ functions to Compute Efficiency for opt ---------------------------------------------------------------

    def ComputeEffAlgo_DiTauJetPNet(self, par):

        #load all events that pass denominator Selection
        events = self.get_events()

        L1Taus = get_L1Taus(events)
        L1Jets = get_L1Jets(events)
        Jets = get_Jets(events)
        GenTaus = get_GenTaus(events)
        GenJets = get_GenJets(events)
        n_min = 2

        # Selection of L1/Gen and Jets objects without PNET WP
        L1Tau_IsoTau26er2p1_mask = L1Tau_IsoTau26er2p1_selection(events)
        L1Tau_IsoTau26er2p1L2NN_mask = L1Tau_IsoTau26er2p1_mask & L1Tau_L2NN_selection_DiTauJet(events)
        L1Jet_Jet55_mask = L1Jet_Jet55_selection(events, L1Tau_IsoTau26er2p1_mask)
        DiTauJet_mask = Jet_selection_DiTauJet(events, par, apply_PNET_WP = False)
        DiTauJet_Jet_mask = Jet_selection_DiTauJet_Jets(events, DiTauJet_mask)
        GenTau_mask = hGenTau_selection(events)
        GenJet_mask = hGenJet_selection(events)
        # at least n_min L1tau/ recoJet and 1 L1jet/jet should pass + L1 trigger
        DiTauJet_evt_mask = (
            (ak.sum(DiTauJet_mask, axis=-1) >= n_min) & (ak.sum(L1Tau_IsoTau26er2p1L2NN_mask, axis=-1) >= n_min) & # taus
            (ak.sum(DiTauJet_Jet_mask, axis=-1) >= 1) & (ak.sum(L1Jet_Jet55_mask, axis=-1) >= 1) # jet
        )
        # matching
        L1Taus_DoubleTauJet = get_selL1Taus(L1Taus, L1Tau_IsoTau26er2p1L2NN_mask, n_min_taus = n_min)
        L1Jets_DoubleTauJet = get_selL1Jets(L1Jets, L1Jet_Jet55_mask, n_min_jets = 1)
        Jets_DoubleTauJet = Jets[DiTauJet_mask]
        Jets_DoubleTauJet_Jet = Jets[DiTauJet_Jet_mask]
        GenTaus_DoubleTauJet = GenTaus[GenTau_mask]
        GenJets_DoubleTauKet = GenJets[GenJet_mask]

        matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTauJet, Jets_DoubleTauJet, GenTaus_DoubleTauJet)
        # at least n_min GenTau should match L1Tau/Taus
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= n_min)

        matchingGenjets_mask = matching_Genjets(L1Jets_DoubleTauJet, Jets_DoubleTauJet_Jet, GenJets_DoubleTauJet)
        # at least 1 GenJet should match L1Jet/Jets
        evt_mask_matching = evt_mask_matching & (ak.sum(matchingGenjets_mask, axis=-1) >= 1)

        DiTauJet_evt_mask = DiTauJet_evt_mask & evt_mask_matching
        N_den = len(events[DiTauJet_evt_mask])
        print(f"Number of events in denominator: {N_den}")

        # Numerator: only need to require PNET WP on the taus, nothing additional on the jets
        # Selection of L1/Gen and Jets objects with PNET WP
        DiTauJet_mask = Jet_selection_DiTauJet(events, par, apply_PNET_WP = True)
        # at least 1 L1tau/ Jet/ GenTau should pass
        DiTauJet_evt_mask = (
            (ak.sum(DiTauJet_mask, axis=-1) >= n_min) & (ak.sum(L1Tau_IsoTau26er2p1L2NN_mask, axis=-1) >= n_min) & # taus
            (ak.sum(DiTauJet_Jet_mask, axis=-1) >= 1) & (ak.sum(L1Jet_Jet55_mask, axis=-1) >= 1) # jet
        )

        # matching
        # no need to match jets, as they are already included in the denominator
        Jets_DoubleTauJet = Jets[DiTauJet_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTauJet, Jets_DoubleTauJet, GenTaus_DoubleTauJet)
        # at least n_min GenTau should match L1Tau/Taus
        evt_mask_matching = evt_mask_matching & (ak.sum(matchingGentaus_mask, axis=-1) >= n_min)

        DiTauJet_evt_mask = DiTauJet_evt_mask & evt_mask_matching
        N_num = len(events[DiTauJet_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

    def ComputeEffAlgo_HLTDoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60(self):

        #load all events that pass denominator Selection
        events = self.get_events()

        L1Taus = get_L1Taus(events)
        L1Jets = get_L1Jets(events)
        Jets = get_Jets(events)
        GenTaus = get_GenTaus(events)
        GenJets = get_GenJets(events)
        n_min = 2

        # Selection of L1/Gen and Jets objects without PNET WP
        L1Tau_IsoTau26er2p1_mask = L1Tau_IsoTau26er2p1_selection(events)
        L1Tau_IsoTau26er2p1L2NN_mask = L1Tau_IsoTau26er2p1_mask & L1Tau_L2NN_selection_DiTauJet(events)
        L1Jet_Jet55_mask = L1Jet_Jet55_selection(events, L1Tau_IsoTau26er2p1_mask)
        DiTauJet_mask = Jet_selection_DiTauJet(events)
        DiTauJet_Jet_mask = Jet_selection_DiTauJet_Jets(events, DiTauJet_mask)
        GenTau_mask = hGenTau_selection(events)
        GenJet_mask = hGenJet_selection(events)
        # at least n_min L1tau/ recoJet and 1 L1jet/jet should pass + L1 trigger
        DiTauJet_evt_mask = (
            (ak.sum(DiTauJet_mask, axis=-1) >= n_min) & (ak.sum(L1Tau_IsoTau26er2p1L2NN_mask, axis=-1) >= n_min) & # taus
            (ak.sum(DiTauJet_Jet_mask, axis=-1) >= 1) & (ak.sum(L1Jet_Jet55_mask, axis=-1) >= 1) # jet
        )
        # matching
        L1Taus_DoubleTauJet = get_selL1Taus(L1Taus, L1Tau_IsoTau26er2p1L2NN_mask, n_min_taus = n_min)
        L1Jets_DoubleTauJet = get_selL1Jets(L1Jets, L1Jet_Jet55_mask, n_min_jets = 1)
        Jets_DoubleTauJet = Jets[DiTauJet_mask]
        Jets_DoubleTauJet_Jet = Jets[DiTauJet_Jet_mask]
        GenTaus_DoubleTauJet = GenTaus[GenTau_mask]
        GenJets_DoubleTauJet = GenJets[GenJet_mask]

        matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTauJet, Jets_DoubleTauJet, GenTaus_DoubleTauJet)
        # at least n_min GenTau should match L1Tau/Taus
        evt_mask_matching = (ak.sum(matchingGentaus_mask, axis=-1) >= n_min)

        matchingGenjets_mask = matching_Genjets(L1Jets_DoubleTauJet, Jets_DoubleTauJet_Jet, GenJets_DoubleTauJet)
        # at least 1 GenJet should match L1Jet/Jets
        evt_mask_matching = evt_mask_matching & (ak.sum(matchingGenjets_mask, axis=-1) >= 1)

        DiTauJet_evt_mask = DiTauJet_evt_mask & evt_mask_matching
        N_den = len(events[DiTauJet_evt_mask])
        print(f"Number of events in denominator: {N_den}")

        # Selection of L1/Gen and Jets objects with Deeptau WP
        DiTauJet_mask = Tau_selection_DiTauJet(events, apply_DeepTau_WP = True)
        # at least 2 L1tau/ Jet/ GenTau and 1 L1jet / Jet / GenJet should pass
        DiTauJet_evt_mask = (
            (ak.sum(DiTauJet_mask, axis=-1) >= n_min) & (ak.sum(L1Tau_IsoTau26er2p1L2NN_mask, axis=-1) >= n_min) & # taus
            (ak.sum(DiTauJet_Jet_mask, axis=-1) >= 1) & (ak.sum(L1Jet_Jet55_mask, axis=-1) >= 1) # jet
        )

        # matching
        # no need to match jets, as they are already included in the denominator
        Jets_DoubleTauJet = Jets[DiTauJet_mask]
        matchingGentaus_mask = matching_Gentaus(L1Taus_DoubleTauJet, Jets_DoubleTauJet, GenTaus_DoubleTauJet)
        # at least n_min GenTau should match L1Tau/Taus
        evt_mask_matching = evt_mask_matching & (ak.sum(matchingGentaus_mask, axis=-1) >= n_min)

        DiTauJet_evt_mask = DiTauJet_evt_mask & evt_mask_matching
        N_num = len(events[DiTauJet_evt_mask])
        print(f"Number of events in numerator: {N_num}")
        print('')
        return N_den, N_num

