import json
import os
import pandas as pd

p = "/eos/home-j/jleonhol/www/PNetAtHLT/PNetAtHLT/Optimisation/result/"

p = "/eos/home-j/jleonhol/www/PNetAtHLT/2024_v1/PNetAtHLT/OptimisationDiTau/result/"

l = []
for f in os.listdir(p):
    with open(p + f) as f:
        l.append(json.load(f))

df = pd.DataFrame(l)

df.to_pickle("results_ditau.pickle")
