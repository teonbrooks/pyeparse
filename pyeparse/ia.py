# -*- coding: utf-8 -*-
# Authors: Teon Brooks <teon.brooks@gmail.com>
#
# License: BSD (3-clause)

from copy import deepcopy
import numpy as np
from scipy.optimize import fmin_slsqp
import warnings
from collections import OrderedDict

from ._event import Discrete
from .viz import plot_epochs
from .utils import pupil_kernel
from ._fixes import string_types, nanmean, nanstd
from .parallel import parallel_func
from ._baseraw import read_raw, _BaseRaw


class InterestAreas(OrderedDict):
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
    dep : str
        Dependent measure. 'fix' for fixations, 'sac' for saccades.
    ignore_missing : bool
        If True, do not warn if no events were found.

    Returns
    -------
    epochs : instance of Epochs
        The epoched dataset.
        Trial x IAS x fixations/saccades
    """
    def __init__(self, raw, ias, event_id=42, dep='fix'):
        if isinstance(raw, str):
            raw = read_raw(raw)
        assert isinstance(raw, _BaseRaw)
        trial_ids = raw.find_events('TRIALID', 1)
        self.n_trials = n_trials = trial_ids.shape[0]
        self.n_ias = n_ias = ias.shape[0]
        last = trial_ids[-1].copy()
        last[0] = str(int(last[0]) + 10000)
        msg = last[-1].split()
        last[-1] = ' '.join([msg[0], str(int(msg[-1]) + 1)])
        trial_ids = np.vstack((trial_ids, last))
        _trial_starts = [int(trial_ids[i][0]) for i, _ in enumerate(trial_ids[:-1])]
        _trial_ends = [int(trial_ids[i+1][0]) for i, _ in enumerate(trial_ids[:-1])]
        _trials = zip(_trial_starts, _trial_ends)
        _trial_durations = [end - start for start, end in _trials]
        
        ias_ = [[list() for ia in range(n_ias)] for _ in range(n_trials)]

        fix_order = [list() for _ in range(n_trials)]
        for fix in raw.discrete['fixations']:
            eye, stime, etime, x_pos, y_pos = fix
            for jj, trial in enumerate(_trials):
                tstart, tend = trial
                if tstart < stime * 1000 < tend:
                    fix_order[jj].append(int(stime * 1000))                                
                    # RECTANGLE id left top right bottom [label]
                    for ii, ia in enumerate(ias):
                        _, _, ia_left, ia_top, ia_right, ia_bottom, _ = ia
                        if int(ia_left) < int(x_pos) < int(ia_right) \
                            and int(ia_top) < int(y_pos) < int(ia_bottom):
                            ias_[jj][ii].append(list(fix))

        ia_labels = list(ias[:, -1])
        # ordered dict
        temp = [dict(zip(ia_labels, [np.array(ia) for ia in trial])) for trial in ias_]
        super.__init__(zip(range(n_trials), temp))

    def __repr__(self):
        return '<IA | {0} Trials x {0} IAs>'.format(self.n_trials, self.n_ias)

    def __getitem__(self, idx):
        if isinstance(idx, string_types):
            if not idx in self[0]:
                raise KeyError("'%s' is not found." % idx)
            return [d[idx] for d in self.values()]
        elif isinstance(idx, int):
            return self.values()[idx]
        elif isinstance(idx, slice):
            return self.values()[idx]
        else:
            raise TypeError('index must be an int, string, or slice')

# Look for trial start and stop
# Gather all the data for the trial
# First fixation, then work on saccade


def read_ia(filename):
    ia = open(filename).readlines()
    idx = [i for i, line in enumerate(ia) if line.startswith('Type') ]
    if len(idx) > 1:
        raise IOError('Too many headers provided in this file.')
    ias = ia[idx[0]+1:]
    ias = np.array([line.split() for line in ias])

    return ias
