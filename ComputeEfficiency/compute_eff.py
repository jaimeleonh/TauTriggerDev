from helpers import compute_ratio_witherr, load_cfg_file
import os 
import uproot
import awkward as ak

config = load_cfg_file()
HLT_name = config['HLT']['HLTname']
MCDataFolderNames = list(filter(None, (x.strip() for x in config['DATA']['MCDataFolderNames'].splitlines())))

#print("!!!!!!!!!!!!!!!!!CAUTION!!!!!!!!!!!!!!!!!")
#MCDataFolderNames = [MCDataFolderNames[0]]


PNet_mode = config['OPT']['PNet_mode']
if PNet_mode == 'false':
    PNetMode = False
    output_path = os.path.join(config['DATA']['result_eff'], HLT_name, HLT_name)
else:
    PNetMode = True
    PNetparam = [float(config['OPT']['PNet_t1']), float(config['OPT']['PNet_t2']), float(config['OPT']['PNet_t3'])]
    output_path = os.path.join(config['DATA']['result_eff'], HLT_name, f'PNetTresh_{config["OPT"]["PNet_t1"]}_{config["OPT"]["PNet_t2"]}_{config["OPT"]["PNet_t3"]}')

N_den = 0
N_num = 0
for i in range(len(MCDataFolderNames)):
    fileName = os.path.join(output_path, f'{MCDataFolderNames[i]}.root') 
    file = uproot.open(fileName)

    TausDen_pt = ak.flatten(file['TausDen']['Tau_pt'].array())
    TausNum_pt = ak.flatten(file['TausNum']['Tau_pt'].array())

    N_den += len(TausDen_pt)
    N_num += len(TausNum_pt)


eff, err_low, err_up = compute_ratio_witherr(N_num, N_den)

if HLT_name == 'HLT_DoubleMediumDeepTauPFTauHPS35_L2NN_eta2p1':
    print(f"Total number of hadronic GenTaus with with vis. pt <= 20 and eta< 2.1: {N_den}")
    if PNetMode:
        print(f"Total number of GenTaus matching with Jet/L1Tau passing DiTau path with Pnet param {PNetparam}: {N_num}")
    else:
        print(f"Total number of GenTaus matching with Tau/L1Tau passing {HLT_name} requirements: {N_num}")

elif HLT_name == 'HLT_DoubleMediumDeepTauPFTauHPS30_L2NN_eta2p1_PFJet60':
    print(f"Total number of hadronic GenTaus with with vis. pt <= 20 and eta< 2.1: {N_den}")
    if PNetMode:
        print(f"Total number of GenTaus matching with Jet/L1Tau passing DiTauJet path with Pnet param {PNetparam}: {N_num}")
    else:
        print(f"Total number of GenTaus matching with Tau/L1Tau passing {HLT_name} requirements: {N_num}")

elif HLT_name == 'HLT_LooseDeepTauPFTauHPS180_L2NN_eta2p1_v3':
    print(f"Total number of hadronic GenTaus with with vis. pt <= 20 and eta< 2.1: {N_den}")
    if PNetMode:
        print(f"Total number of GenTaus matching with Jet/L1Tau passing SingleTau path with Pnet param {PNetparam}: {N_num}")
    else:
        print(f"Total number of GenTaus matching with Tau/L1Tau passing {HLT_name} requirements: {N_num}")

print('Computed Eff: ')
print(f"Eff : {eff}")
print(f"Eff_up : {err_low}")
print(f"Eff_down : {err_up}")


