import numpy as np
import matplotlib.pyplot as plt
from icecream import ic
from glob import glob

if __name__ == "__main__":
    for fname in glob('results/*.npz'):
        results = np.load(fname)
        ic(fname, results)

        target = results['target']
        target_values = results['target_values']
        data = results['results']
        other_values = results['other_values']
        filter_configs = results['filter_configs']
        x_log = results['x_log'] if 'x_log' in results else True
        multithreaded = results['multithreaded']
        
        fig, ax = plt.subplots(2, sharex=True, sharey=True, figsize=(10, 6))
        for idx_method, method in enumerate(['Conditioning', 'Application']):
            ax[idx_method].plot(target_values, data[:,idx_method,:], marker='o', label=filter_configs)

            ax[idx_method].set_ylabel(f'Processing rate\n{method} [Sps]')
            if x_log:
                ax[idx_method].set_xscale('log')
            ax[idx_method].set_yscale('log')
            ax[idx_method].grid()

        ax[0].legend(ncol=3, loc=(0.1, 1.15))
        ax[1].set_xlabel(target)
        p1 = ('Multi-thread' if multithreaded else 'Single-thread')
        p2 = ', '.join([f'{k}={v}' for k, v in other_values])
        ax[0].set_title(p1 + ' ' + p2)

    plt.show()
