from doepy import build
import numpy as np

################################################################################
################################################################################
def bin_arch_params(input_mat, param_dict):
    output_mat = np.zeros(input_mat.values.shape, dtype=int)
    # print(output_mat.shape)
    idx_0_keys = np.digitize(input_mat.values[:, 0],  param_dict['l1d_ways'], right=True)
    idx_1_keys = np.digitize(input_mat.values[:, 1],  param_dict['l1i_ways'], right=True)
    idx_2_keys = np.digitize(input_mat.values[:, 2],  param_dict['l2_ways'], right=True)
    idx_3_keys = np.digitize(input_mat.values[:, 3],  param_dict['l1d_size'], right=True)
    idx_4_keys = np.digitize(input_mat.values[:, 4],  param_dict['l1i_size'], right=True)
    idx_5_keys = np.digitize(input_mat.values[:, 5],  param_dict['l2_size'], right=True)
    idx_6_keys = np.digitize(input_mat.values[:, 6],  param_dict['cacheline'], right=True)

    output_mat[:, 0] = np.array(param_dict['l1d_ways'])[idx_0_keys]
    output_mat[:, 1] = np.array(param_dict['l1i_ways'])[idx_1_keys]
    output_mat[:, 2] = np.array(param_dict['l2_ways'])[idx_2_keys]
    output_mat[:, 3] = np.array(param_dict['l1d_size'])[idx_3_keys]
    output_mat[:, 4] = np.array(param_dict['l1i_size'])[idx_4_keys]
    output_mat[:, 5] = np.array(param_dict['l2_size'])[idx_5_keys]
    output_mat[:, 6] = np.array(param_dict['cacheline'])[idx_6_keys]

    return output_mat
################################################################################
################################################################################
def get_samples_arch_deffe(app_params, n_samples = 100, sampling_method = 'lhs'):

    valid_methods = ['frac_fact_res', 'plackett_burman', 'box_behnken',
                    'central_composite_ccf', 'central_composite_cci', 'central_composite_ccc',
                    'lhs', 'space_filling_lhs', 'random_k_means', 'maximin', 'halton', 'uniform_random']

    if sampling_method not in valid_methods:
        print("ERROR: INVALID SAMPLING METHOD")
        return

    param_dict = {'l1d_ways':[0, 8],
    'l1i_ways':[0, 8],
    'l2_ways':[0, 8],
    'l1d_size':[0, 64],
    'l1i_size':[0, 64],
    'l2_size':[0, 256],
    'cacheline':[0, 128]}

    param_vals = {'l1d_ways': [1, 2, 4, 8],
    'l1i_ways': [1, 2, 4, 8],
    'l2_ways': [1, 2, 4, 8],
    'l1d_size': [1, 2, 4, 8, 16, 32, 64],
    'l1i_size': [1, 2, 4, 8, 16, 32, 64],
    'l2_size': [1, 2, 4, 8, 16, 32, 64, 128, 256],
    'cacheline': [16, 32, 64, 128]}

    if sampling_method == 'frac_fact_res':
        sample_mat = build.frac_fact_res(param_dict)

    if sampling_method == 'plackett_burman':
        sample_mat = build.plackett_burman(param_dict)

    if sampling_method == 'box_behnken':
        sample_mat = build.box_behnken(param_dict)

    if sampling_method == 'central_composite_ccf':
        sample_mat = build.central_composite(param_dict, face='ccf')

    if sampling_method == 'central_composite_cci':
        sample_mat = build.central_composite(param_dict, face='cci')

    if sampling_method == 'central_composite_ccc':
        sample_mat = build.central_composite(param_dict, face='ccc')

    if sampling_method == 'lhs':
        sample_mat = build.lhs(param_dict, num_samples=n_samples)

    if sampling_method == 'space_filling_lhs':
        sample_mat = build.space_filling_lhs(param_dict, num_samples=n_samples)

    if sampling_method == 'random_k_means':
        sample_mat = build.random_k_means(param_dict, num_samples=n_samples)

    if sampling_method == 'maximin':
        sample_mat = build.maximin(param_dict, num_samples=n_samples)

    if sampling_method == 'halton':
        sample_mat = build.halton(param_dict, num_samples=n_samples)

    if sampling_method == 'uniform_random':
        sample_mat = build.uniform_random(param_dict, num_samples=n_samples)

    sample_mat = bin_arch_params(sample_mat, param_vals) # integer-valued discrete samples

    output_mat = np.zeros((len(app_params)*sample_mat.shape[0], sample_mat.shape[1]+1))
    # output_mat = []
    start_idx = 0
    inc_idx = len(sample_mat)
    for app_tmp in app_params:
        tmp_ones = app_tmp*np.ones((len(sample_mat),), dtype=int)
        output_mat[start_idx:start_idx+inc_idx, :sample_mat.shape[1]] = sample_mat
        output_mat[start_idx:start_idx+inc_idx, -1] = tmp_ones
        start_idx += inc_idx


    return output_mat        
    # print(output_mat)
    # print(output_mat.shape)
################################################################################
get_samples_arch_deffe([10, 100, 1000])
