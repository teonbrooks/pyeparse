# -*- coding: utf-8 -*-
# Authors: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

from copy import copy
import numpy as np
try:
    from pandas import DataFrame, concat
except ImportError:
    raise ImportError('Pandas is required for Reading.')

from .interest_areas import InterestAreas


"""Terminology
The definitions for the Reading class methods are derived from [citation].
"""
terms = {
         'Gaze': 'Whether a fixation is in an interest area that is being '
                 'fixated for the first time, and it is the max position in a '
                 'sequence.',
         'First Fixation Duration': 'null',
         'Gaze Duration': 'null',
         'Go-Past Duration': 'null',
         'Dwell Time': 'null',
         }


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
    trial_msg : str
        The identifying tag for trial. Default is 'TRIALID'.

    Returns
    -------
    Reading : instance of Reading class
        An Interest Area report with methods for reading time measurements.
        Trial x IAS x fixations/saccades

    """
    def __init__(self, raw, ias, ia_labels=None, depmeas='fix',
                 trial_msg='TRIALID'):

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

    def __repr__(self):
        ee, ii = self.n_trials, self.n_ias

        return '<Reading | {0} Trials x {1} IAs>'.format(ee, ii)

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
                if gaze_ias[meas['fix_pos'].astype(int)] == 0:
                    gaze_ias[meas['fix_pos'].astype(int)] = 1
                    gaze[idx] = 1
                    first_fix[idx] = 1
                elif gaze[idx - 1] == 1 and \
                    meas['fix_pos'] == prev_meas['fix_pos']:
                    gaze[idx] = 1

        return max_pos, gaze, first_fix

    def get_dwell_time(self, ia):
        data = self._data[self._data['fix_pos'] == ia]
        data = data.groupby(by=['trial'], as_index=False).sum()

        return data

    def get_gaze_duration(self, ia, first_fix=True):
        data = self._data[self._data['gaze'] == 1]
        data = data[data['fix_pos'] == ia]
        data = data.groupby(by=['trial'], as_index=False).sum()
        data = data.reset_index(drop=True)
        if first_fix:
            ffd = self._data[self._data['first_fix'] == 1]
            ffd = ffd[ffd['fix_pos'] == ia]['dur']
            ffd = ffd.reset_index(drop=True)
            columns = list(data.columns)
            columns.append('ffd')
            data = concat((data, ffd), axis=1)
            data.columns = columns

        return data

    def get_go_past(self, ia):
        data = self._data[self._data['max_pos'] == ia]
        data = data.groupby(by=['trial'], as_index=False).sum()

        return data
