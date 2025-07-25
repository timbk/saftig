"""Plotting for data throughput profiling of the filter implementations."""

from glob import glob
import numpy as np
import matplotlib.pyplot as plt
from icecream import ic

MARKER_MAPPING = {
    "WF": "o",
    "UWF": "^",
    "LMS": "x",
    "LMS_C": "P",
    "PolyLMS": "v",
}


def main():
    """Plotting routine"""
    for fname in glob("results/*.npz"):
        results = np.load(fname)
        ic(fname, results)
        fname_core = ".".join(fname.split("/")[-1].split(".")[:-1])

        target = results["target"]
        target_values = results["target_values"]
        data = results["results"]
        other_values = results["other_values"]
        filter_configs = results["filter_configs"]
        filter_names = results["filter_names"]
        x_log = results["x_log"] if "x_log" in results else True
        multithreaded = results["multithreaded"]
        platform = results["platform"]
        git_hash = results["git_hash"]
        cpu_load = results["cpu_load"]

        _fig, ax = plt.subplots(2, sharex=True, sharey=True, figsize=(10, 8))
        for idx_method, method in enumerate(["Conditioning", "Application"]):
            for filter_name, dataset, label in zip(
                filter_names, data[:, idx_method, :].T, filter_configs
            ):
                ax[idx_method].plot(
                    target_values,
                    dataset,
                    marker=MARKER_MAPPING[filter_name],
                    label=label,
                )

            ax[idx_method].set_ylabel(f"Processing rate\n{method} [Sps]")
            if x_log:
                ax[idx_method].set_xscale("log")
            ax[idx_method].set_yscale("log")
            ax[idx_method].grid()
            ax[idx_method].set_ylim(1e4, 2e8)

        ax[0].legend(ncol=3, loc=(0.1, 1.3))
        ax[1].set_xlabel(target)
        ax[0].set_title(
            f'{"Multi-thread" if multithreaded else "Single-thread"} {", ".join([f"{k}={v}" for k, v in other_values])}\n'
            + f"git: {git_hash}, {platform}\n"
            + f"cpu before: {cpu_load[0]*100:.1f}% during: {cpu_load[1]*100:.1f}% after: {cpu_load[2]*100:.1f}%"
        )
        plt.subplots_adjust(top=0.75)

        plt.savefig(f"plots/{fname_core}.png")


if __name__ == "__main__":
    main()

    # plt.show()
