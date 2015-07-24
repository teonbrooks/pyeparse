# Authors: Denis Engemann <denis.engemann@gmail.com>
#
# License: BSD (3-clause)

import os.path as op

import pyeparse as pp
import numpy as np

path = op.dirname(__file__)
fname = op.join(path, '../pyeparse/tests/data/test_raw.edf')

raw = pp.read_raw(fname)

# visualize initial calibration
raw.plot_calibration(title='5-Point Calibration')

# create heatmap
raw.plot_heatmap(start=3., stop=60.)

# find events and epoch data
events = raw.find_events('SYNCTIME', event_id=1)
tmin, tmax, event_id = -0.5, 1.5, 1
epochs = pp.Epochs(raw, events=events, event_id=event_id, tmin=tmin,
                   tmax=tmax)

# access pandas data frame and plot single epoch
import pylab as pl
pl.figure()
pl.plot(epochs[3].get_data('xpos')[0], epochs[3].get_data('ypos')[0])

# iterate over and access numpy arrays.
# find epochs withouth loss of tracking / blinks
print(len([e for e in epochs if not np.isnan(e).any()]))

pl.figure()
pl.title('Superimposed saccade responses')
n_trials = 12  # first 12 trials
for epoch in epochs[:n_trials]:
    pl.plot(epochs.times * 1e3, epoch[0].T)
pl.show()

time_mask = epochs.times > 0
times = epochs.times * 1e3

pl.figure()
pl.plot(times[time_mask], epochs.data[0, 0, time_mask])
pl.title('Post baseline saccade (X, pos)')
pl.show()

# plot single trials
epochs.plot(picks=['xpos'], draw_discrete='saccades')
