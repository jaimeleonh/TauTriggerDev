HLTnameDiTau = 'HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1'
HLTnameSingleTau = 'HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3'
HLTnameDiTauJet = 'HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60'

rate_deepTau_DiTau = 56
rate_deepTau_SingleTau = 18.1
rate_deepTau_DiTauJet = 18.1

WP_params_SingleTau = {
    'Tight': {'t1': '1.0',
              't2': '0.99',
              'rate': 12.9},
    'Medium': {'t1': '0.98',
               't2': '0.98',
              'rate': 17.9},
    'Loose': {'t1': '0.92',
              't2': '0.92',
              'rate': 28.3},
    # 'MaxEff': {'t1': '0.0',
               # 't2': '0.0',
              # 'rate': 779.4},
}

WP_params_DiTau = {
    'Tight': {'t1': '0.56',
              't2': '0.54',
              'rate': 45.79},
    'Medium': {'t1': '0.57',
               't2': '0.44',
              'rate': 55.69},
    # 'MaxEff': {'t1': '0.0',
               # 't2': '0.0',
              # 'rate': 4135.3},
}

WP_params_DiTauJet = {
    'Tight': {'t1': '0.65',
              't2': '0.41',
              'rate': 17.85},
}
