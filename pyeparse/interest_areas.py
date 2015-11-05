# -*- coding: utf-8 -*-
# Authors: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

from copy import copy
import numpy as np
try:
    from pandas import DataFrame, concat, notnull
except ImportError:
    raise ImportError('Pandas is required for InterestAreas.')

from ._fixes import string_types, OrderedDict
from ._baseraw import read_raw, _BaseRaw


terms = {
         'index': 'The fixation order.',
         'eye': '`0` corresponds to the left eye, `1` corresponds to the right,'
                ' and `2` corresponds to binocular tracking.',
        stime: 'Start time of fixation.',
        etime: 'End time of fixation.',
        axp: 'Average x-position',
        ayp: 'Average y-position',
        dur: 'Duration of fixation: `etime` - `stime`.',
        fix_pos: 'Index of the interest area.',
        trial: 'Trial Number.',
        max_pos: 'Maximum position in a sequence.',
        gaze: 'Boolean of the first time(s) fixating on a given item in a '
               'sequence. Must also be the max position in the sequence.',
        first_fix: 'Boolean of the very first fixation on a given item in a '
                   'sequence. Must also be the max position in the sequence.'
        }

class InterestAreaReport(object):
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
    InterestAreaReport : an instance of InterestAreaReport
        An Interest Area report. Fixations or Saccades are limited to the areas
        described in `ias`. The fix/sacc are labeled in a given region
        per trial.
        Info is stored for Trial x IAS x fixations/saccades
    """
    def __init__(self, raw, ias, ia_labels=None, depmeas='fix'):

        if isinstance(raw, string_types):
            raw = read_raw(raw)
        elif not isinstance(raw, _BaseRaw):
            raise TypeError('raw must be Raw instance of filename, not %s'
                            % type(raw))
        if isinstance(ias, string_types):
            ias = read_ia(ias)

        trial_ids = raw.find_events('TRIALID', 1)
        self.n_trials = n_trials = trial_ids.shape[0]
        self.n_ias = n_ias = ias.shape[0]
        last = trial_ids[-1].copy()
        last[0] = str(int(last[0]) + 10000)
        trial_ids = np.vstack((trial_ids, last))
        t_starts = [int(trial_ids[i][0]) for i, _ in enumerate(trial_ids[:-1])]
        t_ends = [int(trial_ids[i+1][0]) for i, _ in enumerate(trial_ids[:-1])]
        self._trials = trials = zip(t_starts, t_ends)
        self.trial_durations = [end - start for start, end in trials]

        if depmeas == 'fix':
            data = raw.discrete['fixations']
            labels = ['eye', 'stime', 'etime', 'axp', 'ayp']
        elif depmeas == 'sac':
            data = raw.discrete['saccades']
            labels = ['eye', 'stime', 'etime', 'sxp', 'syp',
                      'exp', 'eyp', 'pv']
        else:
            raise NotImplementedError
        data = DataFrame(data)
        data.columns = labels

        # adding new columns, initializing to np.nan
        # fix_pos is the IA number
        fix_pos = np.ones((len(data), 1)) * np.nan
        trial_no = np.ones((len(data), 1)) * np.nan

        for idx, meas in data.iterrows():
            for jj, trial in enumerate(trials):
                tstart, tend = trial
                if tstart < meas['stime'] * 1000 < tend:
                    trial_no[idx] = jj
                    # RECTANGLE id left top right bottom [label]
                    if depmeas == 'fix':
                        for ii, ia in enumerate(ias):
                            _, _, ia_left, ia_top, ia_right, ia_bottom, _ = ia
                            if int(ia_left) < int(meas['axp']) < int(ia_right) \
                            and int(ia_top) < int(meas['ayp']) < int(ia_bottom):
                                fix_pos[idx] = ii
                                break
                    elif depmeas == 'sac':
                        for ii, ia in enumerate(ias):
                            _, _, ia_left, ia_top, ia_right, ia_bottom, _ = ia
                            if int(ia_left) < int(meas['sxp']) < int(ia_right) \
                            and int(ia_top) < int(meas['syp']) < int(ia_bottom):
                                fix_pos[idx] = ii
                    break

        dur = data['etime'] - data['stime']
        data = map(DataFrame, [data, dur, fix_pos, trial_no])
        labels.extend(['dur', 'fix_pos', 'trial'])
        data = concat(data, axis=1)
        data.columns = labels

        # drop all fixations outside of the interest areas
        data = data[notnull(data['fix_pos'])]
        data = data.reset_index()
        data.fix_pos = data.fix_pos.astype(int)

        if ia_labels is not None:
            ias[:, -1] = ias_labels
        # DataFrame
        self._data = data
        self._ias = dict([(ii[-1], int(ii[1])) for ii in ias])
        self.ia_labels = sorted(self._ias, key=self._ias.get)

    def __repr__(self):
        ee, ii = self.n_trials, self.n_ias

        return '<InterestAreas | {0} Trials x {1} IAs>'.format(ee, ii)

    def __len__(self):
        return len(self._data)

    @property
    def shape(self):
        return (self.n_trials, self.n_ias)

    def __getitem__(self, idx):
        if isinstance(idx, string_types):
            if idx not in self._ias:
                raise KeyError("'%s' is not found." % idx)
            data = self._data[self._data['fix_pos'] == self._ias[idx]]
            return data
        elif isinstance(idx, int):
            idx = self._data['trial'] == idx
            data = self._data[idx]
            return data
        elif isinstance(idx, slice):
            inst = copy(self)
            inst._data = self._data[idx]
            inst.n_trials = len(inst._data)
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

class Reading(InterestAreas):
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
    Reading : instance of Reading class
        An Interest Area report with methods for reading time measurements.
        Trial x IAS x fixations/saccades

    """
    def __init__(self, raw, ias, ia_labels=None, depmeas='fix'):

        super(Reading, self).__init__(
            raw=raw, ias=ias, ia_labels=ia_labels, depmeas=depmeas)

        labels = self._data.columns.get_values()
        max_pos, gaze, first_fix = self._define_gaze()
        data = map(DataFrame, [max_pos, gaze, first_fix])
        data.insert(0, self._data)
        labels = np.hstack((labels, ['max_pos', 'gaze', 'first_fix']))
        data = concat(data, axis=1)
        data.columns = labels
        self._data = data

    def _define_max_pos(self):
        data = self._data
        # # create a max position using lag
        # max_pos is the maximum IA visited
        max_pos = np.ones((len(data), 1)) * np.nan
        max_pos[0] = data.iloc[0]['fix_pos']
        for idx, meas in enumerate(data.iterrows()):
            _, meas = meas
            if meas['trial'] != data.iloc[idx - 1]['trial']:
                max_pos[idx] = meas['fix_pos']
            else:
                if meas['fix_pos'] > max_pos[idx - 1]:
                    max_pos[idx] = meas['fix_pos']
                else:
                    max_pos[idx] = max_pos[idx - 1]

        return max_pos

    def _define_gaze(self):
        max_pos = self._define_max_pos()
        data = self._data
        # initializing
        gaze = np.zeros((len(self._data), 1), int)
        gaze[0] = 1
        gaze_ias = np.zeros((self.n_ias, 1))
        ref_trial = data.iloc[0]['trial']
        first_fix = np.zeros((len(self._data), 1), int)

        for idx in range(1, len(data)):
            meas = data.iloc[idx]
            prev_meas = data.iloc[idx - 1]
            if meas['trial'] > ref_trial:
                gaze_ias = np.zeros((self.n_ias, 1), int)
                ref_trial = meas['trial']
            if meas['fix_pos'] == max_pos[idx]:
                if gaze_ias[meas['fix_pos']] == 0:
                    gaze_ias[meas['fix_pos']] = 1
                    gaze[idx] = 1
                    first_fix[idx] = 1
                elif gaze[idx - 1] == 1 and \
                    meas['fix_pos'] == prev_meas['fix_pos']:
                    gaze[idx] = 1
                # else:
                #     gaze[idx] = 0

        return max_pos, gaze, first_fix

    def get_first_fix(self, ia):
        data = self._data[self._data['first_fix'] == 1]
        data = data[data['fix_pos'] == ia]

        return data

    def get_dwell_time(self, ia):
        data = self._data[self._data['fix_pos'] == ia]
        data = data.groupby(by=['trial']).sum()

        return data

    def get_gaze_duration(self, ia):
        data = self._data[self._data['gaze'] == 1]
        data = data[data['fix_pos'] == ia]
        data = data.groupby(by=['trial']).sum()

        return data

    def get_go_past(self, ia):
        data = self._data[self._data['max_pos'] == ia]
        data = data.groupby(by=['trial']).sum()

        return data
