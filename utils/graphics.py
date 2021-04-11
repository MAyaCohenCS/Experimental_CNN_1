import matplotlib.pyplot as plt
import math
import pickle

'''
How to use graphics module:
1) import it:
# import utils.graphics as grph

2) initialize it. pick a general title for the plot:
# perf = grph.Perf(desc='super_title')

3) initialize a session. (AKA pick a title for the current subplot):
# pref.new_session('subplot title')

4) collect data. attach a new single measurement to your active session:
# perf.read_data_point(train_data_point, test_data_point)

* repeat (3), (4) as much as you like

5) print graphics! it will be automatically saved to doc/figures/raw as well, 
so pick a meaningful filename for it. file suffix (.png) is automatically generated.
if you add the 'show=True' flag it will also pop out when it's ready:
# pref.plot_performance('filename', show=True)
'''

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
            ax_i.set(xlabel='train progress \n(batch number)', ylabel='Loss')
            ax_i.label_outer()
            ax_i.set_box_aspect(1)

            if i == 0:
                handles, labels = ax_i.get_legend_handles_labels()
                fig.legend(handles, labels, loc='right')

        plt.savefig('./doc/figures/raw/' + filename)

        # pref object for later use (graphics corrections, etc.)
        perf_backup = open('./doc/figures/raw_data/' + filename + '.pickle', 'wb')
        pickle.dump(self, perf_backup)
        perf_backup.close()

        if show:
            plt.show()


