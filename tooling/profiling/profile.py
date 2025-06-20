import saftig as sg
import numpy as np

DEBUG = False
IGNORE_FILTER_OPTIONS = {'coefficient_clipping', 'step_scale'}

# filter, additional_filter_config, skip_conditioning
FILTER_CONFIGURATIONS = [
    (sg.WienerFilter, {}, False),
    (sg.UpdatingWienerFilter, {'context_pre': 1000}, True),
    (sg.UpdatingWienerFilter, {'context_pre': 2000}, True),
    (sg.LMSFilter, {'normalized': True, 'coefficient_clipping': 10}, False),
    (sg.LMSFilter, {'normalized': False, 'coefficient_clipping': 10, 'step_scale': 0.001}, False),
    (sg.PolynomialLMSFilter, {'order': 1, 'coefficient_clipping': 10}, False),
    (sg.PolynomialLMSFilter, {'order': 2, 'coefficient_clipping': 10}, False),
    (sg.PolynomialLMSFilter, {'order': 3, 'coefficient_clipping': 10}, False),
]

if DEBUG:
    FILTER_CONFIGURATIONS = [
            (sg.WienerFilter, {}),
            (sg.LMSFilter, {'normalized': True}),
            (sg.UpdatingWienerFilter, {'context_pre': 1000}, True),
    ]

def additional_filter_config_to_str(conf):
    if len(conf) == 0:
        return ''
    return '('+', '.join(f'{k}={v}' for k, v in conf.items() if k not in IGNORE_FILTER_OPTIONS)+')'
filter_configuration_strings = [f'{fc.filter_name} {additional_filter_config_to_str(settings)}' for fc, settings, _ignore_conditioning in FILTER_CONFIGURATIONS ]


def run_profiling(config, n_samples, n_filter, n_channel):
    filters = map(lambda x: x[0], config)
    additional_settings = map(lambda x: x[1], config)
    skip_conditioning = list(map(lambda x: x[2], config))

    results = sg.measure_runtime(filters, n_samples, n_filter=n_filter, n_channel=n_channel, additional_filter_settings=additional_settings)
    results = np.array(results)
    results[0,skip_conditioning] = np.nan
    return results

def profiling_scan_n_filter(values = [10, 30, 100, 300, 1000]):
    filter_configs = FILTER_CONFIGURATIONS
    n_samples = int(1e4)
    n_channel = 1

    results = []
    for n_filter in values:
        print(f'n_filter = {n_filter}')
        res = run_profiling(filter_configs, n_samples, n_filter, n_channel)
        results.append(n_samples/np.array(res))
    return results

def main():
    n_filter_values = [10, 30, 100, 300, 1000]
    n_filter_scan_results = profiling_scan_n_filter(n_filter_values)
    np.savez('profiling_results.npz', n_filter_scan = n_filter_scan_results, n_filter_values=n_filter_values, filter_configurations=filter_configuration_strings)

if __name__ == "__main__":
    main()
