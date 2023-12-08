import pandas as pd
import json
import numpy as np

if __name__ == '__main__':
    def plot(xaxis, yaxis, parameters, x_title, y_title, min_x, max_x, min_y, max_y, save_path, data=None):
        import matplotlib
        matplotlib.use("PDF")
        from matplotlib import pyplot as plt
        plt.figure()
        ax = plt.subplot()
        #if min_x and max_x:
        #    ax.set_xlim(min_x, max_x)
        #if min_y and max_y:
        #    ax.set_ylim(min_y, max_y)
        try:
            ax = data.plot(x=xaxis, y=yaxis, kind="scatter", ax=ax)
            if min_x and max_x:
               ax.set_xlim(min_x, max_x)
            if min_y and max_y:
               ax.set_ylim(min_y, max_y)
        except:
            plt.scatter(xaxis, yaxis, marker="o")
        for (x, y, label) in zip(df[xaxis], df[yaxis], parameters):
            continue
            plt.annotate(label, # this is the text
                 (x, y), # this is the point to label
                 textcoords="offset points", # how to position the text
                 xytext=(0, 10), # distance from text to points (x,y)
                 ha='center', # horizontal alignment can be left, right or center
                 size=5)
        plt.xlabel(x_title)
        plt.ylabel(y_title)

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

    with open("results.json") as f:
        d = json.load(f)

    df = pd.DataFrame(d)
    df.convert_dtypes()


    # plot("rate", "efficiency", df["params"], "Rate (Hz)", "Efficiency", 17.80, 18.20, 0.84, 0.86, "plot.pdf", data=df)
    #plot("rate", "efficiency", df["params"], "Rate (Hz)", "Efficiency", 17.50, 18.20, 0.84, 0.86, "plot.pdf", data=df)
    plot("rate", "efficiency", df["params"], "Rate (Hz)", "Efficiency", 0., 1., 0., 1., "plot.pdf", data=df)
    #plot([15], [0.95], [[1,2]], "Rate (Hz)", "Efficiency", 10, 20, 0.9, 1., "plot.pdf")
