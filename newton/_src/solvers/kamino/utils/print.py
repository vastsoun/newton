###########################################################################
# KAMINO: Utilities: Console Output
###########################################################################

import sys
import time
import numpy as np


###
# Console output
###

def printmatrix(x: np.ndarray, name: str = None):
    if name:
        print(name, ":")
    for row in x:
        print(' '.join(map(lambda x: "{:.3f}\t".format(x), row)))


def printvector(x: np.ndarray, name: str = None):
    if name:
        print(name, ":")
    print(' '.join(map(lambda x: "{:.3f}\t".format(x), x)))


def print_progress_bar(iteration, total, start_time, length=40, prefix='', suffix=''):
    """
    Display a progress bar with ETA and estimated FPS.

    Args:
        iteration (int) : Current iteration
        total (int) : Total iterations
        start_time (float) : Start time from time.time()
        length (int) : Character length of the bar
        prefix (str) : Prefix string
        suffix (str) : Suffix string
    """
    elapsed = time.time() - start_time
    progress = iteration / total
    filled_length = int(length * progress)
    bar = '█' * filled_length + '-' * (length - filled_length)

    # Estimated Time of Arrival
    if iteration > 0:
        eta = elapsed / iteration * (total - iteration)
        eta_str = time.strftime('%H:%M:%S', time.gmtime(eta))
        fps = iteration / elapsed
        fps_str = f'{fps:.2f} fps'
    else:
        eta_str = 'Calculating...'
        fps_str = '-- fps'

    line_reset = ' ' * 120
    sys.stdout.write(f'\r{line_reset}')
    sys.stdout.write(f'\r{prefix} |{bar}| {iteration}/{total} ETA: {eta_str} ({fps_str}) {suffix}')
    sys.stdout.flush()

    if iteration == total:
        sys.stdout.write('\n')
