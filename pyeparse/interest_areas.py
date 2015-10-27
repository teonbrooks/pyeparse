# -*- coding: utf-8 -*-
# Authors: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

from copy import copy
import numpy as np
try:
    from pandas import DataFrame, concat
except ImportError:
    raise ImportError('Pandas is required for InterestAreas.')

from ._fixes import string_types, OrderedDict
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
    ia_labels : None | list of str
        Interest area name. If None, the labels from the interest area file
        will be used.
    depmeas : str
        Dependent measure. 'fix' for fixations, 'sac' for saccades.

    Returns
    -------
    epochs : instance of Epochs
        The epoched dataset.
        Trial x IAS x fixations/saccades
    """
    def __init__(self, raw, ias, ia_labels=None, depmeas='fix'):

        # TEMP
        if depmeas != 'fix':
            raise NotImplementedError

        if isinstance(raw, string_types):
            raw = read_raw(raw)
        elif not isinstance(raw, _BaseRaw):
            raise TypeError('raw must be Raw instance of filename, not %s'
                            % type(raw))
        if isinstance(ias, string_types):
            ias = read_ia(ias)

        trial_ids = raw.find_events('TRIALID', 1)
        self.n_epochs = n_epochs = trial_ids.shape[0]
        self.n_ias = n_ias = ias.shape[0]
        last = trial_ids[-1].copy()
        last[0] = str(int(last[0]) + 10000)
        trial_ids = np.vstack((trial_ids, last))
        t_starts = [int(trial_ids[i][0]) for i, _ in enumerate(trial_ids[:-1])]
        t_ends = [int(trial_ids[i+1][0]) for i, _ in enumerate(trial_ids[:-1])]
        self._trials = trials = zip(t_starts, t_ends)
        self.trial_durations = [end - start for start, end in trials]

        if depmeas == 'fix':
            depmeas = raw.discrete['fixations']
            labels = ['eye', 'stime', 'etime', 'axp', 'ayp']
            depmeas = DataFrame(depmeas)
            # adding new columns
            # fix_pos is the IA number
            fix_pos = np.ones((len(depmeas), 1)) * np.nan
            # max_pos is the maximum IA visited
            max_pos = np.ones((len(depmeas), 1)) * np.nan
            trial_no = np.ones((len(depmeas), 1)) * np.nan

            for idx, meas in depmeas.iterrows():
                for jj, trial in enumerate(trials):
                    tstart, tend = trial
                    # max_ia is incorrect atm
                    max_ia = 0
                    if tstart < meas['stime'] * 1000 < tend:
                        trial_no[idx] = jj
                        # RECTANGLE id left top right bottom [label]
                        for ii, ia in enumerate(ias):
                            _, _, ia_left, ia_top, ia_right, ia_bottom, _ = ia
                            if int(ia_left) < int(meas['axp']) < int(ia_right) \
                            and int(ia_top) < int(meas['ayp']) < int(ia_bottom):
                                fix_pos[idx] = ii
                                # TO-DO: think about cases where fix outside
                                # interest areas
                                if ii > max_ia:
                                    max_ia = ii
                                    max_pos[idx] = max_ia
                                else:
                                    max_pos[idx] = max_ia
                                break
                        break
            dur = depmeas['etime'] - depmeas['stime']
            depmeas = map(DataFrame, [depmeas, dur, fix_pos, max_pos, trial_no])
            labels.extend(['dur', 'fix_pos', 'max_pos', 'trial_no'])
            depmeas = concat(depmeas, axis=1)
            depmeas.columns = labels

            # # think of a way to fix the max_pos problem
            # for idx, meas in depmeas.iterrows():
            #     if idx == 0:
            #         pass
            #     if meas['fix_pos'] <

        # adapt from above
        elif depmeas == 'sac':
            depmeas = raw.discrete['saccades']
            labels = ['eye', 'stime', 'etime', 'sxp', 'syp',
                      'exp', 'eyp', 'pv']
            depmeas = DataFrame(depmeas)
            for _, meas in depmeas.iterrows():
                for jj, trial in enumerate(trials):
                    tstart, tend = trial
                    if tstart < meas['stime'] * 1000 < tend:
                        fix_order[jj].append(int(meas['stime'] * 1000))
                        # RECTANGLE id left top right bottom [label]
                        for ii, ia in enumerate(ias):
                            _, _, ia_left, ia_top, ia_right, ia_bottom, _ = ia
                            if int(ia_left) < int(meas['sxp']) < int(ia_right) \
                            and int(ia_top) < int(meas['syp']) < int(ia_bottom):
                                ias_[jj][ii].append(meas)
            labels.append('order')
            n_meas = len(labels)
            for jj in range(n_epochs):
                for ii in range(n_ias):
                    if isinstance(ias_[jj][ii], list) and \
                        len(ias_[jj][ii]) == 0:
                        ias_[jj][ii] = DataFrame(np.ones((1, n_meas)) *
                                                        np.nan, columns=labels)
                    else:
                        for kk, fix in enumerate(ias_[jj][ii]):
                            fix['order'] = kk
                            ias_[jj][ii][kk] = fix
                        ias_[jj][ii] = DataFrame(ias_[jj][ii])

        if ia_labels is not None:
            ias[:, -1] = ias_labels
        # DataFrame
        self._data = depmeas
        self._ias = dict([(ii[-1], int(ii[1])) for ii in ias])
        self.ia_labels = sorted(self._ias, self._ias.get)

    def __repr__(self):
        return '<IA | {0} Trials x {1} IAs>'.format(self.n_epochs, self.n_ias)

    def __len__(self):
        return len(self._data)

    @property
    def shape(self):
        return (self.n_epochs, self.n_ias)

    def __getitem__(self, idx):
        if isinstance(idx, string_types):
            if idx not in self._ias:
                raise KeyError("'%s' is not found." % idx)
            data = self._data[self._data['fix_pos'] == self._ias[idx]]
            return data
        elif isinstance(idx, int):
            data = self._data[idx:idx + 1]
            return data
        elif isinstance(idx, slice):
            inst = copy(self)
            inst._data = self._data[idx]
            inst.n_epochs = len(inst._data)
            return inst
        else:
            raise TypeError('index must be an int, string, or slice')


def read_ia(filename):
    with open(filename) as FILE:
        ia = FILE.readlines()
    idx = [i for i, line in enumerate(ia) if line.startswith('Type')]
    if len(idx) > 1:
        raise IOError('Too many headers provided in this file.')
    ias = ia[idx[0]+1:]
    ias = np.array([line.split() for line in ias])

    return ias

class ReadingMixin(object):
    def _define_gaze(self):
        fix_num = trial['order']
        maxword = np.maximum.accumulate(trial['order'])
        gaze = fix_num <= maxword

    def get_first_fix(self, ia, idx=None):

        return self
    def get_go_past(self, ia, idx=None):
        fixs = self[ia]
        if idx is None:
            idx = np.arange(self.shape[0])
        fixs = fixs[idx]
        trials = np.unique(fixs['trial'])
        for ii in trials:
            trial = trials[trials['trial'] == ii]
            maxword = np.maximum.accumulate(trial['order'])
