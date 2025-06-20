import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

if __name__ == "__main__":
    results = np.load('profiling_results.npz')

    data = results['n_filter_scan']
    n_filter = results['n_filter_values']
    filter_confs = results['filter_configurations']
    
    fig, ax = plt.subplots(2, sharex=True, sharey=True, figsize=(10, 6))
    for idx_method, method in enumerate(['Conditioning', 'Application']):
        ax[idx_method].plot(n_filter, data[:,idx_method,:], marker='o', label=filter_confs)

        ax[idx_method].set_xlabel('n_filter')
        ax[idx_method].set_ylabel(f'Processing rate\n{method} [Sps]')
        ax[idx_method].set_xscale('log')
        ax[idx_method].set_yscale('log')
        ax[idx_method].grid()
    ax[0].legend(ncol=3, loc=(0.1, 1))

    plt.show()
