# Authors: Denis Engemann <d.engemann@fz-juelich.de>
#
# License: BSD (3-clause)

import numpy as np

def plot_calibration(raw, title='Calibration', show=True):
    """Visualize calibration

    Parameters
    ----------
    raw : instance of pylinkparse raw
        The raw object to be visualized
    title : str
        The title to be displayed.
    show : bool
        Whether to show the figure or not.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The resulting figure object
    """
    import pylab as pl
    cal = raw.info['validation']

    px, py = cal[['point-x', 'point-y']].values.T
    dx, dy = cal[['diff-x', 'diff-y']].values.T

    fig = pl.figure()
    pl.title(title)
    pl.scatter(px, py, color='gray')
    pl.scatter(px - dx, py - dy, color='red')
    if show:
        pl.show()
    return fig


def plot_heatmap(xdata, ydata, width, height, cmap=None,
                 show=True):
    """ Plot heatmap of X/Y positions on canvas, e.g., screen

    Parameters
    ----------
    xdata : array-like
        The X position data to be visualized.
    ydata : array-like
        The Y position data to be visualized.
    width : int
        The canvas width.
    height : int
        The canvas height.

    Returns
    -------
    fig : instance of matplotlib.figure.Figure
        The resulting figure object
    canvas : ndarray (width, height)
        The canvas including the gaze data.
    """
    import pylab as pl
    if cmap is None:
        cmap = 'RdBu_r'

    canvas = np.zeros((width, height))
    data = np.c_[xdata, ydata]
    inds = data[(data[:, 0] > 0) &
                (data[:, 1] > 0) &
                (data[:, 0] < width) &
                (data[:, 1] < height)].astype('i4')
    for x, y in inds:
        canvas[x, y] += 1
    fig = pl.figure()
    pl.imshow(canvas, extent=[0, width, 0, height],
              cmap=cmap, aspect='auto', origin='lower')

    if show:
        pl.show()
    return fig, canvas
