import matplotlib.pyplot as plt
import math

TRAIN_COLOR = '#4D5E76'
TEST_COLOR = '#ECCA50'


class Perf:
    def __init__(self, desc, sampling_rate=500):
        self.curr_train_series = None
        self.curr_test_series = None
        self.previous_data_series = []
        self.description = desc
        self.subtitles = []
        self.sampling_rate = sampling_rate

    def read_data_point(self, train_datapoint, test_datapoint):
        self.curr_train_series += [train_datapoint]
        self.curr_test_series += [test_datapoint]

    def close_data_series(self):
        new_series = (tuple(self.curr_train_series), tuple(self.curr_test_series))
        self.previous_data_series += [new_series]

        self.curr_train_series = None
        self.curr_test_series = None

    def new_session(self, session_title):
        if self.curr_train_series or self.curr_test_series:
            self.close_data_series()

        self.curr_train_series = []
        self.curr_test_series = []

        self.subtitles += [session_title]

    def plot_performance(self, filename, show=False, rows=1):
        if self.curr_train_series or self.curr_test_series:
            self.close_data_series()

        cols = math.ceil(len(self.previous_data_series) / rows)
        fig, ax = plt.subplots(rows, cols, sharex='all', sharey='all')
        fig.suptitle(self.description)

        for i, data_series in enumerate(self.previous_data_series):
            train_data, test_data = data_series

            if cols == 1 and rows == 1:
                ax_i = ax
            else:
                ax_i = ax[i]

            ax_i.plot(train_data, label='Train', color=TRAIN_COLOR)
            ax_i.plot(test_data, label='Test', color=TEST_COLOR)
            ax_i.set_title(self.subtitles[i])
            ax_i.set(xlabel='train progress (batch number)', ylabel='Loss')
            ax_i.label_outer()
            ax_i.set_box_aspect(1)

            if i == 0:
                handles, labels = ax_i.get_legend_handles_labels()
                fig.legend(handles, labels, loc='right')

        plt.savefig('./doc/figures/' + filename)
        if show:
            plt.show()

#  usage example:
# import utils.graphics as graphics
# perfs = [graphics.Perf([10, 7, 3], [9, 4, 6], 'subtitle1')] *5
# title = 'test'
# plotname = 'test'
# graphics.plot_performance(perfs, title, plotname, True, cols=5)
