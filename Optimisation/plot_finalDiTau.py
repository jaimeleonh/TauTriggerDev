import pandas as pd
import json
import numpy as np

if __name__ == '__main__':
    def plot(xaxis, yaxis, parameters, x_title, y_title, min_x, max_x, min_y, max_y, save_path, data=None, params_to_mark=[]):
        params_to_mark.append(("deeptau",))
        import matplotlib
        matplotlib.use("PDF")
        from matplotlib import pyplot as plt
        plt.figure()
        ax = plt.subplot()
        #if min_x and max_x:
        #    ax.set_xlim(min_x, max_x)
        #if min_y and max_y:
        #    ax.set_ylim(min_y, max_y)
        
        # data_final = data[data["params"].astype(str) == "[26, 0.64, 0.46]"]
        # print(data_final)
        # try:
            # ax = data_final.plot(x=xaxis, y=yaxis, kind="scatter", ax=ax)
            # if min_x and max_x:
               # ax.set_xlim(min_x, max_x)
            # if min_y and max_y:
               # ax.set_ylim(0., 1.)
        # except:
            # ax = plt.scatter(xaxis, yaxis, marker="o")
            # # if min_x and max_x:
               # # ax.set_xlim(min_x, max_x)
            # # if min_y and max_y:
        # ax.set_xlim(50, 65)
        # ax.set_ylim(0.2, 0.8)

        for (x, y) in zip(data_final[xaxis], data_final[yaxis]):
            # continue
            plt.annotate("PNet", # this is the text
                 (x, y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0, 10), # distance from text to points (x,y)
                 ha='center', # horizontal alignment can be left, right or center
                 size=10)
        plt.xlabel(x_title)
        plt.ylabel(y_title)

        # plt.plot(dd["rate"], dd["efficiency"], "*")
        # plt.annotate("DeepTau", # this is the text
             # (dd["rate"], dd["efficiency"]), # this is the point to label
             # textcoords="offset points", # how to position the text
             # xytext=(0, 10), # distance from text to points (x,y)
             # ha='center', # horizontal alignment can be left, right or center
             # size=10)

        # plt.plot(data["rate_noditaujet"][0], data["eff_noditaujet"][0], "*")
        # plt.annotate("PNet, no DiTau+Jet", # this is the text
             # (data["rate_noditaujet"][0], data["eff_noditaujet"][0]), # this is the point to label
             # textcoords="offset points", # how to position the text
             # xytext=(0, 10), # distance from text to points (x,y)
             # ha='center', # horizontal alignment can be left, right or center
             # size=10)

        # plt.plot(dd["noditaujet_rate"], dd["noditaujet_eff"], "*")
        # plt.annotate("DeepTau, no DiTau+Jet", # this is the text
             # (dd["noditaujet_rate"], dd["noditaujet_eff"]), # this is the point to label
             # textcoords="offset points", # how to position the text
             # xytext=(0, 10), # distance from text to points (x,y)
             # ha='center', # horizontal alignment can be left, right or center
             # size=10)

        x_text=0.05
        y_text=0.9
        plt.text(x_text, 1.02, "CMS", fontsize='large', fontweight='bold',
            transform=ax.transAxes)
        upper_text = "private work"
        plt.text(x_text + 0.1, 1.02, upper_text, transform=ax.transAxes)
        # text = [self.dataset.process.label.latex, self.category.label.latex]
        # for t in text:
            # plt.text(x_text, y_text, t, transform=ax.transAxes)
            # y_text -= 0.05


        plt.savefig(save_path)
        plt.close('all')

    # with open("results_tri_optimised.json") as f:
        # d = json.load(f)


    # import glob
    # read_files = glob.glob("/eos/home-j/jleonhol/www/PNetAtHLT/PNetAtHLT/Optimisation/result/*.json")
    # d = []                                                                                                                                                                                                    
    # for f in read_files:                                                                                                                                                                                                
        # with open(f, "rb") as infile:                                                                                                                                                                                     
           # d.append(json.load(infile))

    df = pd.read_pickle("/afs/cern.ch/work/j/jleonhol/private/TauTriggerDev/results_ditau.pickle")   

    # df = df[(df['eff'] > 0.679) & (df['eff'] < 0.68) & (df['rate'] > 60 ) & (df['rate'] < 61)]
    # df = df[(df["params"] == [26, 0.64, 0.46])]
    

    # df = pd.DataFrame(d)
    # df.convert_dtypes()

    # with open("results.json") as f:
        # d_noditaujet = json.load(f)

    # # # # with open("results_deeptau.json") as f:
        # # # # dd = json.load(f)

    # # # # results = {
        # # # # "method": [
            # # # # "DeepTau",
            # # # # "DeepTau (no DiTauJet)",
            # # # # "PNet (no DiTauJet)",
        # # # # ],
        # # # # "rate": [
            # # # # dd["rate"],
            # # # # dd["noditaujet_rate"],
            # # # # df["rate_noditaujet"][0],
        # # # # ],
        # # # # "efficiency": [
            # # # # dd["efficiency"],
            # # # # dd["noditaujet_eff"],
            # # # # df["eff_noditaujet"][0],
        # # # # ]
    # # # # }
    # # # # results = pd.DataFrame(results)
    # # # # print(results)

    # plot("rate", "efficiency", df["params"], "Rate (Hz)", "Efficiency", 17.80, 18.20, 0.84, 0.86, "plot.pdf", data=df)
    # plot("rate", "eff", df["params"], "Rate (Hz)", "Efficiency", 60, 61, 0.675, 0.68, "plot_tri_pt_optimised_cut.pdf", data=df)
    # plot("rate", "eff", df["params"], "Rate (Hz)", "Efficiency", 60, 61, 0.679, 0.68, "plot_tri_pt_optimised_cut.pdf", data=df)
    
    plot("rate", "eff", df["params"], "Rate (Hz)", "Efficiency", 0., 100., 0., 1., "plot_ditau_optimised_cut.pdf", data=df)
    #plot([15], [0.95], [[1,2]], "Rate (Hz)", "Efficiency", 10, 20, 0.9, 1., "plot.pdf")
