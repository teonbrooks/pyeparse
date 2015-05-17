# -*- coding: utf-8 -*-
# Authors: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

from copy import copy, deepcopy
import numpy as np
from collections import OrderedDict

from ._fixes import string_types
from ._baseraw import read_raw, _BaseRaw


class InterestAreas(object):
    """ Create interest area summaries for Raw

    Parameters
    ----------
    raw : filename | instance of Raw
        The filename of the raw file or a Raw instance to create InterestAreas
        from.
    ias : str | ndarray (n_ias, 7)
        The interest areas. If str, this is the path to the interest area file.
        Only .ias formats are currently supported. If array, number of row must
        be number of interest areas. The columns are as follows:
        # RECTANGLE id left top right bottom [label]
    ias_names : None | list of str
        Interest area name. If None, the areas will be indexed (0, n-1).
    event_id : int | dict
        The event ID to use. Can be a dict to supply multiple event types
        by name.
    depmeas : str
        Dependent measure. 'fix' for fixations, 'sac' for saccades.

    Returns
    -------
    epochs : instance of Epochs
        The epoched dataset.
        Trial x IAS x fixations/saccades
    """
    def __init__(self, raw, ias, event_id=42, depmeas='fix'):
        if isinstance(raw, string_types):
            raw = read_raw(raw)
        if not isinstance(raw, _BaseRaw):
            raise TypeError('raw must be Raw instance of filename, not %s'
                            % type(raw))

        trial_ids = raw.find_events('TRIALID', 1)
        self.n_trials = n_trials = trial_ids.shape[0]
        self.n_ias = n_ias = ias.shape[0]
        last = trial_ids[-1].copy()
        last[0] = str(int(last[0]) + 10000)
        msg = last[-1].split()
        last[-1] = ' '.join([msg[0], str(int(msg[-1]) + 1)])
        trial_ids = np.vstack((trial_ids, last))
        t_starts = [int(trial_ids[i][0]) for i, _ in enumerate(trial_ids[:-1])]
        t_ends = [int(trial_ids[i+1][0]) for i, _ in enumerate(trial_ids[:-1])]
        self._trials = trials = zip(t_starts, t_ends)
        self.trial_durations = [end - start for start, end in trials]

        ias_ = [[list() for ia in range(n_ias)] for _ in range(n_trials)]
        fix_order = [list() for _ in range(n_trials)]

        if depmeas == 'fix':
            depmeas = raw.discrete['fixations']
            labels = ['eye', 'stime', 'etime', 'axp', 'ayp']
            n_meas = len(labels)
            for meas in depmeas:
                eye, stime, etime, xp, yp = meas
                meas = np.array(meas)
                for jj, trial in enumerate(trials):
                    tstart, tend = trial
                    if tstart < stime * 1000 < tend:
                        fix_order[jj].append(int(stime * 1000))                                
                        # RECTANGLE id left top right bottom [label]
                        for ii, ia in enumerate(ias):
                            _, _, ia_left, ia_top, ia_right, ia_bottom, _ = ia
                            if int(ia_left) < int(xp) < int(ia_right) \
                                and int(ia_top) < int(yp) < int(ia_bottom):
                                ias_[jj][ii] = meas
            for jj in range(n_trials):
                for ii in range(n_ias):
                    if isinstance(ias_[jj][ii], list):
                        ias_[jj][ii] = np.rec.fromarrays(np.ones(n_meas) *
                                                         np.nan, names=labels)
        elif depmeas == 'sac':
            depmeas = raw.discrete['saccades']
            labels = ['eye', 'stime', 'etime', 'sxp', 'syp',
                      'exp', 'eyp', 'pv']
            n_meas = len(labels)
            for meas in depmeas:
                eye, stime, etime, sxp, syp, exp, eyp, pv = meas
                meas = np.array(meas)
                for jj, trial in enumerate(trials):
                    tstart, tend = trial
                    if tstart < stime * 1000 < tend:
                        fix_order[jj].append(int(stime * 1000))                                
                        # RECTANGLE id left top right bottom [label]
                        for ii, ia in enumerate(ias):
                            _, _, ia_left, ia_top, ia_right, ia_bottom, _ = ia
                            if int(ia_left) < int(sxp) < int(ia_right) \
                                and int(ia_top) < int(syp) < int(ia_bottom):
                                ias_[jj][ii] = meas
            for jj in range(n_trials):
                for ii in range(n_ias):
                    if isinstance(ias_[jj][ii], list):
                        ias_[jj][ii] = np.rec.fromarrays(np.ones(n_meas) *
                                                         np.nan, names=labels)

        ia_labels = list(ias[:, -1])
        # ordered dict
        temp = [IA(zip(ia_labels, trial)) for trial in ias_]
        self._data = temp

    def __repr__(self):
        return '<IA | {0} Trials x {1} IAs>'.format(self.n_trials, self.n_ias)

    def __len__(self):
        return len(self._data)

    @property
    def shape(self):
        return (self.n_trials, self.n_ias)

    def __getitem__(self, idx):
        if isinstance(idx, string_types):
            if not idx in self._data[0]:
                raise KeyError("'%s' is not found." % idx)
            return np.hstack([datum[idx] for datum in self._data])
        elif isinstance(idx, int):
            return self._data[idx]
        elif isinstance(idx, slice):
            inst = copy(self)
            inst._data = self._data[idx]
            inst.n_trials = len(inst._data)
            return inst
        else:
            raise TypeError('index must be an int, string, or slice')


class IA(OrderedDict):
    def __init__(self, entry):
        super(IA, self).__init__(entry)

    def __repr__(self):
        return '<Interest Areas: %s>' % ', '.join(self.keys())


def read_ia(filename):
    ia = open(filename).readlines()
    idx = [i for i, line in enumerate(ia) if line.startswith('Type') ]
    if len(idx) > 1:
        raise IOError('Too many headers provided in this file.')
    ias = ia[idx[0]+1:]
    ias = np.array([line.split() for line in ias])

    return ias
