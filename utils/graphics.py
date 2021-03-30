import matplotlib.pyplot as plt

TRAIN_COLOR = '#4D5E76'
TEST_COLOR = '#ECCA50'


class Perf:
    def __init__(self, train_data, test_data, desc):
        self.train_data = train_data
        self.test_data = test_data
        self.description = desc


def plot_performance(perf_list, title, plotname, show=False, cols=1, rows=1):
    fig, ax = plt.subplots(rows, cols, sharex='all', sharey='all')
    fig.suptitle(title)

    for i, perf in enumerate(perf_list):
        if cols == 1 and rows == 1:
            ax_i = ax
        else:
            ax_i = ax[i]

        ax_i.plot(perf.train_data, label='Train', color=TRAIN_COLOR)
        ax_i.plot(perf.test_data, label='Test', color=TEST_COLOR)
        ax_i.set_title(perf.description)
        ax_i.set(xlabel='train progress (batch)', ylabel='Loss')
        ax_i.label_outer()
        ax_i.set_box_aspect(1)

        if i == 0:
            handles, labels = ax_i.get_legend_handles_labels()
            fig.legend(handles, labels, loc='right')

    plt.savefig('./doc/figures/' + plotname)
    if show:
        plt.show()

#  usage example:
# perfs = [graphics.Perf([10, 7, 3], [9, 4, 6], 'subtitle1')] *5
# title = 'test'
# plotname = 'test'
# graphics.plot_performance(perfs, title, plotname, True, cols=5)
