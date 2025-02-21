import torch
import h5py as h5
import os
import glob
import numpy as np
import shutil
from skimage.metrics import structural_similarity
from scipy.optimize import curve_fit


from siae import DeepKoopmanAE
import plot_utils


class ModelLogger:
    def __init__(self, save_dir, log_step=1, name_fmt="model_"):
        # delete save file if redundant to avoid conflict
        self.save_dir_name = save_dir
        self.save_dir = os.path.join("saved_models", save_dir)
        self.log_step = log_step
        self.name_fmt = name_fmt
        shutil.rmtree(self.save_dir, ignore_errors=True)
        os.makedirs(self.save_dir, exist_ok=True)

    def log(self, model, epoch, force=False):
        if epoch % self.log_step != 0 and not force:
            return
        self.save(model, epoch)

    def save(self, model, epoch):
        filename = f"{self.name_fmt}{epoch}.pt"
        savepath = os.path.join(self.save_dir, filename)
        torch.save(model.state_dict(), savepath)


def read_h5_data(fn, data_key):
    contents = {}
    with h5.File(os.path.join("data", fn), "r") as df:
        for key in df.keys():
            contents[key] = torch.tensor(df[key], dtype=torch.float32)
    data = contents[data_key]
    data = torch.nan_to_num(data, 0.0)
    return data


def setup_data_from_images(flow_data, pred_steps=1):
    data = torch.flatten(flow_data, start_dim=1)
    X = data[:-pred_steps]
    Y = data[pred_steps:]
    return X, Y


def simple_preprocess_data(data, train_len=500, val_len=500):
    data -= torch.mean(data)
    data /= 6 * torch.std(data)
    orig_shape = data.shape[1:]
    train_data = data[:train_len]
    val_data = data[train_len : train_len + val_len]
    X, Y = setup_data_from_images(train_data)
    X_val, Y_val = setup_data_from_images(val_data)
    return X, Y, X_val, Y_val, orig_shape


def load_flow_past_cylinder(train_len=500, val_len=500):
    data = read_h5_data("dmd_cyl5k.h5", "data")
    X, Y, X_val, Y_val, orig_shape = simple_preprocess_data(data, train_len, val_len)
    return X, Y, X_val, Y_val, orig_shape


def load_noised_flow_past_cylinder(train_len=500, val_len=500):
    data = read_h5_data("noised_cyl_data5k.h5", "n_0_4")
    X, Y, X_val, Y_val, orig_shape = simple_preprocess_data(data, train_len, val_len)
    return X, Y, X_val, Y_val, orig_shape


def load_JHU(train_len=3000, val_len=250, step=1):
    # max 5024
    data = read_h5_data("JHU_isocoarse4.h5", "curl")
    data = data[::step]
    X, Y, X_val, Y_val, orig_shape = simple_preprocess_data(data, train_len, val_len)
    return X, Y, X_val, Y_val, orig_shape


def load_bickley(train_len=200, val_len=100):
    # max 5024
    data = read_h5_data("smsqbickley_cont.h5", "data")
    data = data[::1, ::2, ::2]
    # data = data - torch.mean(data, axis=0)
    X, Y, X_val, Y_val, orig_shape = simple_preprocess_data(data, train_len, val_len)
    return X, Y, X_val, Y_val, orig_shape


def load_SST(train_len=1000, val_len=500):
    # max 1584
    # data = read_h5_data('sst_mean.h5', 'sst_mean')
    data = read_h5_data("sst_mean.h5", "sst_mean")
    data -= torch.mean(data, axis=0)
    X, Y, X_val, Y_val, orig_shape = simple_preprocess_data(data, train_len, val_len)
    return X, Y, X_val, Y_val, orig_shape


def load_SST_gulf(train_len=1000, val_len=500):
    # max 1584
    # data = read_h5_data('sst_mean.h5', 'sst_mean')
    data = read_h5_data("small_sst_mean.h5", "gulf")
    data -= torch.mean(data, axis=0)
    X, Y, X_val, Y_val, orig_shape = simple_preprocess_data(data, train_len, val_len)
    return X, Y, X_val, Y_val, orig_shape


def load_SST_enso(train_len=1000, val_len=500):
    # max 1584
    # data = read_h5_data('sst_mean.h5', 'sst_mean')
    data = read_h5_data("small_sst_mean.h5", "enso")
    data -= torch.mean(data, axis=0)
    X, Y, X_val, Y_val, orig_shape = simple_preprocess_data(data, train_len, val_len)
    return X, Y, X_val, Y_val, orig_shape


def load_TBL04(train_len=500, val_len=500, ten_step=False):
    # max 5k
    if ten_step:
        key = "frames_10step"
    else:
        key = "frames"
    data = read_h5_data("TBL04.h5", key)
    data = torch.nan_to_num(data, 0.0)
    X, Y, X_val, Y_val, orig_shape = simple_preprocess_data(data, train_len, val_len)
    return X, Y, X_val, Y_val, orig_shape


def load_JAXturb(train_len=1000, val_len=500, square_shape=256, stride=8):
    # max 3900
    data = read_h5_data("forced_iso_jax_2.h5", "vorticity")
    data = data[:, ::stride, ::stride]
    print(data.shape)
    # data = data[:,:square_shape,:square_shape]
    X, Y, X_val, Y_val, orig_shape = simple_preprocess_data(data, train_len, val_len)
    return X, Y, X_val, Y_val, orig_shape


def load_noisedJAXturb(train_len=1000, val_len=500, square_shape=256, stride=1):
    # max 3900
    data = read_h5_data("noised_forced_iso_jax01.h5", "vorticity")
    data = data[:, ::stride, ::stride]
    # data = data[:,:square_shape,:square_shape]
    X, Y, X_val, Y_val, orig_shape = simple_preprocess_data(data, train_len, val_len)
    return X, Y, X_val, Y_val, orig_shape


def load_sstanom(train_len=1500, val_len=500):
    # max 2191
    data = read_h5_data("gulf_sst_anom_18_23.h5", "data")
    # data = read_h5_data('gulf_sst_anom_zoom_18_23.h5', 'data')
    data = data[:, ::2, ::2]
    print(data.shape)
    X, Y, X_val, Y_val, orig_shape = simple_preprocess_data(data, train_len, val_len)
    return X, Y, X_val, Y_val, orig_shape


def load_aaolt23(train_len=3000, val_len=500):
    # max 21504
    fn = "ml_0237_M06.h5"
    data_key = "opd"

    contents = {}
    with h5.File(os.path.join("data", fn), "r") as df:
        for key in [data_key]:
            contents[key] = torch.tensor(df[key], dtype=torch.float32)
    data = contents[data_key]
    data = torch.nan_to_num(data, 0.0)
    # return data
    # data = read_h5_data('ml_0237_M06.h5', 'opd')
    print(data.shape)
    X, Y, X_val, Y_val, orig_shape = simple_preprocess_data(data, train_len, val_len)
    return X, Y, X_val, Y_val, orig_shape


def get_model_paths(dir_name):
    search_str = os.path.join("saved_models", dir_name, "*.pt")
    paths = glob.glob(search_str)
    return paths


def get_checkpoints_from_paths(paths):
    """
    take in a list of paths and extract only the epoch number from them,
    return these in a sorted list
    """
    epoch_numbers = []
    for path in paths:
        epoch_number = int(path.split("_")[-1].split(".")[0])
        epoch_numbers.append(epoch_number)
    return epoch_numbers


def get_sorted_path_and_epochs(dir_name):
    paths = get_model_paths(dir_name)
    epoch_nums = get_checkpoints_from_paths(paths)
    paths = np.array(paths)
    epoch_nums = np.array(epoch_nums)
    sorted_indices = np.argsort(epoch_nums)
    paths = paths[sorted_indices]
    epoch_nums = epoch_nums[sorted_indices]
    return paths, epoch_nums


def load_model(base_model, full_path):
    """
    pass in a generic model with same params as saved model
    and load in the weights from the saved state dict
    """
    base_model.load_state_dict(torch.load(full_path))


def load_models_from_dir(dir_name, input_size, output_size, layers, latent_dim):
    paths, epoch_nums = get_sorted_path_and_epochs(dir_name)
    models = {}
    for epoch, path in zip(epoch_nums, paths):
        model = DeepKoopmanAE(input_size, output_size, layers, latent_dim)
        load_model(model, path)
        models[epoch] = model
    return models


def load_dynamics_from_dir(dir_name, input_size, output_size, layers, latent_dim):
    paths, epoch_nums = get_sorted_path_and_epochs(dir_name)
    dynamics = {}
    for epoch, path in zip(epoch_nums, paths):
        model = DeepKoopmanAE(input_size, output_size, layers, latent_dim)
        load_model(model, path)
        dynamics[epoch] = model.linear_dynamics.weight.detach().numpy()
    return dynamics


def gen_preds(model, data, max_steps=1):
    """
    generate predictions without error computation
    """
    input_data = data[:-max_steps]
    preds = torch.zeros(max_steps, input_data.shape[0], input_data.shape[1])
    for i in range(max_steps):
        model.eval()
        with torch.no_grad():
            current_pred = model(input_data)
        preds[i] = current_pred
        input_data = current_pred
    return preds


def evaluated_preds(model, data, max_steps=1):
    input_data = data[:-max_steps]
    preds = torch.zeros(max_steps, input_data.shape[0], input_data.shape[1])
    errors = torch.zeros_like(preds)
    for i in range(max_steps):
        model.eval()
        with torch.no_grad():
            current_pred = model(input_data)
        preds[i] = current_pred
        comparison_data = data[i + 1 : len(data) - max_steps + i + 1]
        errors[i] = comparison_data - current_pred
        input_data = current_pred
    return preds, errors


def gen_latent_preds(model, data, max_steps=1):
    """
    generate predictions in latent space
    """
    model.eval()
    with torch.no_grad():
        encoded_data = model.encoder(data)
    input_data = encoded_data[:-max_steps]
    preds = torch.zeros(max_steps, input_data.shape[0], input_data.shape[1])
    for i in range(max_steps):
        model.eval()
        with torch.no_grad():
            current_pred = model.linear_dynamics(input_data)
        preds[i] = current_pred
        input_data = current_pred
    return preds


def evaluated_latent_preds(model, data, max_steps=1):
    model.eval()
    with torch.no_grad():
        encoded_data = model.encoder(data)
    input_data = encoded_data[:-max_steps]
    preds = torch.zeros(max_steps, input_data.shape[0], input_data.shape[1])
    errors = torch.zeros_like(preds)
    for i in range(max_steps):
        model.eval()
        with torch.no_grad():
            current_pred = model.linear_dynamics(input_data)
        preds[i] = current_pred
        comparison_data = encoded_data[i + 1 : len(data) - max_steps + i + 1]
        errors[i] = comparison_data - current_pred
        input_data = current_pred
    return preds, errors


def evaluate_ssim(preds, data, max_steps=1):
    ssim_vals = np.zeros((max_steps, preds.shape[1]))
    for i in range(max_steps):
        comparison_data = data[i + 1 : len(data) - max_steps + i + 1]
        for j in range(preds.shape[1]):
            ssim_vals[i][j] = ssim(np.array(comparison_data[j]), np.array(preds[i][j]))
    return ssim_vals


def evaluated_preds_metrics(errors):
    mse = torch.mean(torch.mean(torch.sqrt(errors**2), axis=2), axis=1)
    std = torch.std(torch.mean(torch.sqrt(errors**2), axis=2), axis=1)
    return mse, std


def persistence_preds(data, max_steps=1):
    input_data = data[:-max_steps]
    errors = torch.zeros(max_steps, input_data.shape[0], input_data.shape[1])
    for i in range(max_steps):
        comparison_data = data[i + 1 : len(data) - max_steps + i + 1]
        errors[i] = comparison_data - input_data
    return errors


def persistence_ssim(data, max_steps=1):
    input_data = data[:-max_steps]
    ssim_vals = np.zeros((max_steps, input_data.shape[0]))
    for i in range(max_steps):
        comparison_data = data[i + 1 : len(data) - max_steps + i + 1]
        for j in range(input_data.shape[0]):
            ssim_vals[i][j] = ssim(
                np.array(comparison_data[j]), np.array(input_data[j])
            )
    return ssim_vals


def eigen_freq_from_dynamics(dynamics):
    """
    takes in linear operator dynamics and output eigen frequencies
    over the range 0 to pi
    """
    return np.abs(np.log(np.linalg.eig(dynamics)[0]).imag)


def eigen_stability_from_dynamics(dynamics):
    """
    takes in linear operator dynamics and output eigen frequencies
    over the range 0 to pi
    """
    return np.log(np.linalg.eig(dynamics)[0]).real


def spectral_norm(matrix):
    return torch.linalg.norm(matrix, ord=2)


def calc_target_psd(data, fft_stride=1):
    sum_fft = torch.zeros_like(torch.abs(torch.fft.rfft(data[:, 0])))
    for i in range(0, data.shape[1], fft_stride):
        fft = torch.abs(torch.fft.rfft(data[:, i]))
        sum_fft += fft
    avg_fft = sum_fft / data.shape[0]
    return avg_fft


def avg_temporal_psd(data, sample_freq=1, return_first=False):
    """
    faster and more pythonic app averaged psd calc
    """
    # px_frames = np.reshape(np.swapaxes(data, 0, 2), (data.shape[1]*data.shape[2], data.shape[0]))
    px_frames = np.swapaxes(data, 0, 1)
    half_ind = int(px_frames.shape[1] / 2)
    ffts = np.array([np.fft.fft(px_series)[:half_ind] for px_series in px_frames])
    psds = np.abs(ffts).real
    fs = np.fft.fftfreq(px_frames.shape[1], 1 / sample_freq)[:half_ind]
    psds[psds == 0] = np.nan
    app_avg = np.nanmean(psds, axis=0)
    if return_first:
        return fs, app_avg
    return fs[1:], app_avg[1:]


def get_avg_coeffs(model, input_data):
    model.eval()
    with torch.no_grad():
        _, _, zk = model.detailed_forward(input_data)
    avg_zk = torch.mean(zk, axis=0)
    return avg_zk.detach().numpy()


def avg_spatial_psd_OLD(data, orig_shape, sample_freq=1):
    # px_frames = np.swapaxes(data, 0, 1)
    px_frames = data.reshape([data.shape[0], orig_shape[0], orig_shape[1]])
    half_ind = int(px_frames.shape[1] / 2)
    ffts = np.array([np.fft.fft(px_series)[:half_ind] for px_series in px_frames])
    psds = np.abs(ffts).real
    fs = np.fft.fftfreq(px_frames.shape[1], 1 / sample_freq)[:half_ind]
    psds[psds == 0] = np.nan
    app_avg = np.nanmean(psds, axis=0)
    return fs[1:], app_avg[1:]


def avg_spatial_psd(data, orig_shape, sample_freq=1):
    images = data.reshape([data.shape[0], orig_shape[0], orig_shape[1]])

    def radial_average_power_spectrum(image):
        # Determine the size of the square subset
        subset_size = min(image.shape)

        # Calculate the starting index for the subset in each dimension
        start_row = (image.shape[0] - subset_size) // 2
        start_col = (image.shape[1] - subset_size) // 2

        # Extract the square subset
        image = image[
            start_row : start_row + subset_size, start_col : start_col + subset_size
        ]

        fft_result = np.fft.fftshift(np.fft.fft2(image))
        power_spectrum = np.abs(fft_result) ** 2
        frequencies_x = np.fft.fftshift(np.fft.fftfreq(image.shape[0]))
        frequencies_y = np.fft.fftshift(np.fft.fftfreq(image.shape[1]))
        radius = np.sqrt(
            np.meshgrid(frequencies_x, frequencies_y)[0] ** 2
            + np.meshgrid(frequencies_x, frequencies_y)[1] ** 2
        )
        unique_radius = np.unique(radius)
        radial_average = np.zeros_like(unique_radius)
        for i, r in enumerate(unique_radius):
            radial_average[i] = np.mean(power_spectrum[radius == r])
        return unique_radius, radial_average

    radial_avg_spectrum_list = []
    for i in range(images.shape[0]):
        radius, radial_avg_spectrum = radial_average_power_spectrum(images[i])
        radial_avg_spectrum_list.append(radial_avg_spectrum)

    # Compute the mean and standard deviation of the radially averaged power spectra
    mean_spectrum = np.mean(radial_avg_spectrum_list, axis=0)
    # std_spectrum = np.std(radial_avg_spectrum_list, axis=0)
    # TODO replace this with actual frequency
    return range(len(mean_spectrum)), mean_spectrum


def print_arch(model):
    print("Encoder:", model.encoder)
    print("Dynamics:", model.linear_dynamics)
    print("Decoder:", model.decoder)


# def spectral_norm(layer):
#     spectrally_normalized_weight = layer.weight_u.squeeze() / layer.weight_v.squeeze()


def linear_schedule(zero_index, max_index):
    epochs = torch.arange(max_index)
    schedule = torch.nn.functional.relu(torch.tensor(1 - epochs / zero_index))
    return schedule


def low_f_init_func(latent_dim):
    start_eig_fs = np.random.normal(size=(int(latent_dim / 2)))
    start_eig_fs = np.concatenate([-np.abs(start_eig_fs), np.abs(start_eig_fs)])
    start_eig_fs = np.array(
        [
            -1 * (np.pi - (val % np.pi)) if val < 0 else val % np.pi
            for val in start_eig_fs
        ]
    )

    start_eig_gs = np.random.normal(scale=1 / 4, size=(int(latent_dim / 2)))
    start_eig_gs = np.concatenate([-np.abs(start_eig_gs), -np.abs(start_eig_gs)])

    complex_eigs = start_eig_gs + 1.0j * start_eig_fs

    discrete_eigs = np.exp(complex_eigs)

    rand_imag_eigv = np.random.normal(size=(latent_dim, latent_dim))
    rand_real_eigv = np.random.normal(size=(latent_dim, latent_dim))
    rand_eigen_vectors = rand_real_eigv + 1.0j * rand_imag_eigv

    rand_A = np.linalg.multi_dot(
        (rand_eigen_vectors, np.diag(discrete_eigs), np.linalg.pinv(rand_eigen_vectors))
    )

    return torch.tensor(rand_A)

    # custom dynamics initialization
    # def init_dynamics(m):
    #     if isinstance(m, nn.Linear):
    #         custom_weights = utils.low_f_init_func(latent_dim)
    #         m.weight.data.copy_(custom_weights)
    #         # torch.nn.init.xavier_uniform_(m.weight)
    #         # m.bias.data.fill_(0.01)
    # model.linear_dynamics.apply(init_dynamics)


def load_last_model(input_size, output_size, layers, latent_dim, dir_name):
    paths, epoch_nums = get_sorted_path_and_epochs(dir_name)
    model = DeepKoopmanAE(input_size, output_size, layers, latent_dim)
    load_model(model, paths[-1])
    return model


def get_lr_schedule(mode, **kwargs):
    pass


def load_and_plot_model_pred_comp(
    input_size,
    output_size,
    layers,
    latent_dim,
    eval_data,
    max_steps,
    base_dir,
    siae_dir,
    plotter,
    save_fn_append="",
):
    base_model = load_last_model(input_size, output_size, layers, latent_dim, base_dir)
    siae_model = load_last_model(input_size, output_size, layers, latent_dim, siae_dir)
    base_preds, base_errors = evaluated_preds(
        base_model, eval_data, max_steps=max_steps
    )
    siae_preds, siae_errors = evaluated_preds(
        siae_model, eval_data, max_steps=max_steps
    )
    base_mse, base_std = evaluated_preds_metrics(base_errors)
    siae_mse, siae_std = evaluated_preds_metrics(siae_errors)
    per_errors = persistence_preds(eval_data, max_steps=max_steps)
    per_mse, per_std = evaluated_preds_metrics(per_errors)
    plot_utils.plot_pred_step_model_comp_eval(
        base_mse,
        base_std,
        siae_mse,
        siae_std,
        per_mse,
        per_std,
        plotter,
        save_fn_append=save_fn_append,
    )

    ssim_per = persistence_ssim(eval_data, max_steps=max_steps)
    ssim_vals_base = evaluate_ssim(base_preds, eval_data, max_steps=max_steps)
    ssim_vals_siae = evaluate_ssim(siae_preds, eval_data, max_steps=max_steps)
    plot_utils.plot_ssim_model_comp(
        ssim_vals_base, ssim_vals_siae, ssim_per, plotter, save_fn_append=save_fn_append
    )


def load_and_eval_final_model(
    dir_name, input_size, output_size, layers, latent_dim, eval_data, max_steps=10
):
    model = load_last_model(input_size, output_size, layers, latent_dim, dir_name)
    preds, errors = evaluated_preds(model, eval_data, max_steps=max_steps)
    mses, stds = evaluated_preds_metrics(errors)
    return mses, stds
    # return mses[pred_steps], stds[pred_steps]


def load_and_eig_f_final_model(
    dir_name, input_size, output_size, layers, latent_dim, eval_data, pred_steps=[0]
):
    model = load_last_model(input_size, output_size, layers, latent_dim, dir_name)
    dynamics = model.linear_dynamics.weight.detach().numpy()
    eig_fs = eigen_freq_from_dynamics(dynamics)
    return eig_fs


def eig_comp(
    input_size,
    output_size,
    layers,
    latent_dim,
    base_dir,
    siae_dir,
    dmd_op_path,
    plotter,
    save_fn_append="",
):
    pass


def ssim(base, pred):
    (score, diff) = structural_similarity(base, pred, full=True, data_range=2.0)
    return score


def compute_dmd_operator(X, Y, r=None):
    """
    Compute the DMD operator A from snapshots X and Y.

    Parameters:
        X (numpy.ndarray): Matrix of size (n_features, n_snapshots) representing the snapshots at time t.
        Y (numpy.ndarray): Matrix of size (n_features, n_snapshots) representing the snapshots at time t+1.

    Returns:
        numpy.ndarray: The DMD operator A.
    """
    # Compute the SVD of X
    U, Sigma, Vh = np.linalg.svd(X, full_matrices=False)

    # Truncate the SVD to reduce computational cost (optional)
    if r is None:
        r = min(X.shape)
    U_r = U[:, :r]
    Sigma_r_inv = np.diag(1 / Sigma[:r])
    V_rh = Vh[:r, :]

    # Compute the A matrix
    A = U_r.T @ Y @ V_rh.T @ Sigma_r_inv

    return A


def compute_dmd_operator(X, Y, r=None):
    """
    Compute the DMD operator A from snapshots X and Y.

    Parameters:
        X (numpy.ndarray): Matrix of size (n_features, n_snapshots) representing the snapshots at time t.
        Y (numpy.ndarray): Matrix of size (n_features, n_snapshots) representing the snapshots at time t+1.

    Returns:
        numpy.ndarray: The DMD operator A.
    """
    # Compute the SVD of X
    U, Sigma, Vh = np.linalg.svd(X, full_matrices=False)

    # Truncate the SVD to reduce computational cost (optional)
    if r is None:
        r = min(X.shape)
    U_r = U[:, :r]
    Sigma_r_inv = np.diag(1 / Sigma[:r])
    V_rh = Vh[:r, :]

    # Compute the A matrix
    A = U_r.T @ Y @ V_rh.T @ Sigma_r_inv

    return A


# TODO
def linear_error_rejection_approx(n_samples, bw_range):
    """
    construct a linear error rejection function approximation
    """


# TODO
def bw_sweep(pred_series, data, bw_range):
    """
    use a linear error rej approximation to sweep bws for control
    """


def latent_pred_error():
    pass


def spectral_error(data, model):
    def interpolate_psd(eig_fs, f, psd):
        interp_psd = torch.zeros_like(eig_fs)
        for i, freq in enumerate(eig_fs):
            idx = torch.argmin(torch.abs(f - freq))
            if idx == 0:
                interp_psd[i] = psd[0]
            elif idx == len(f) - 1:
                interp_psd[i] = psd[-1]
            else:
                w1 = (f[idx] - freq) / (f[idx] - f[idx - 1])
                w2 = 1 - w1
                interp_psd[i] = w1 * psd[idx - 1] + w2 * psd[idx]
        return interp_psd

    with torch.no_grad():
        sum_fft = torch.zeros_like(torch.abs(torch.fft.rfft(data[:, 0])))
        for i in range(0, data.shape[1]):
            fft = torch.abs(torch.fft.rfft(data[:, i]))
            sum_fft += fft
        data_spectrum = sum_fft / data.shape[0]

        dynamics = model.linear_dynamics.weight
        eigen_vals, eigen_vecs = torch.linalg.eig(dynamics)
        cont_eigen_vals = torch.log(eigen_vals)
        eig_fs = torch.abs(cont_eigen_vals.imag / torch.pi)

        with torch.no_grad():
            _, _, zk = model.detailed_forward(data)
        avg_zk = torch.mean(zk, axis=0)
        eigen_coeffs = torch.abs(torch.matmul(avg_zk, torch.abs(eigen_vecs))) ** 2
        eigen_coeffs = eigen_coeffs * torch.exp(5 * torch.abs(eigen_vals))

        f = torch.linspace(0, 1, len(data_spectrum))
        target = interpolate_psd(eig_fs, f, data_spectrum)

        eigen_coeffs = eigen_coeffs / torch.sum(eigen_coeffs)
        target = target / torch.sum(target)
        error = target - eigen_coeffs
        # error[eig_fs==0] = 0

        reg_loss = torch.sum((error) ** 2) / error.shape[0]
    return reg_loss


# TODO clean up big time
def fourier_error(data, model, pred_steps, fft_stride=1):
    # first compute the target fft
    with torch.no_grad():
        data = data[:pred_steps]
        sum_fft = torch.zeros_like(torch.abs(torch.fft.rfft(data[:, 0])))
        for i in range(0, data.shape[1], fft_stride):
            fft = torch.abs(torch.fft.rfft(data[:, i]))
            sum_fft += fft
        data_spectrum = sum_fft / data.shape[0]
        # return avg_fft

        input_data = model.encoder(data)
        preds = torch.zeros(pred_steps, input_data.shape[0], input_data.shape[1])
        for i in range(pred_steps):
            current_pred = model.linear_dynamics(input_data)
            preds[i] = current_pred
            input_data = current_pred

        sum_fft = torch.zeros_like(torch.abs(torch.fft.rfft(preds[:, 0, 0])))
        for i in range(0, preds.shape[1], fft_stride):
            for j in range(0, preds.shape[2], fft_stride):
                fft = torch.abs(torch.fft.rfft(preds[:, i, j]))
                sum_fft += fft

        # if self.train_data is not None:
        #     target = self.target_psd
        # else:
        #     target = torch.pow(torch.arange(1,len(sum_fft)+1,dtype=torch.float64), -2)
        target = data_spectrum
        target = target / torch.sum(target[1:])

        sum_fft = sum_fft / torch.sum(sum_fft[1:])
        error = target - sum_fft
        error = error[1:]  # skip first value due to computational issues

        reg_loss = torch.sum((error) ** 2) / error.shape[0]
        # reg_loss = torch.sum(torch.abs(error)) / error.shape[0]
    return reg_loss


def encoded_psd(data, model, pred_steps, fft_stride=1, scale_eigs=True):
    data = data[:pred_steps]

    with torch.no_grad():
        dynamics = model.linear_dynamics.weight.detach()

        if scale_eigs:
            eigen_vals, eigen_vecs = torch.linalg.eig(dynamics)
            max_eig_scale = torch.max(torch.abs(eigen_vals))
            # cont_eigen_vals = torch.log(eigen_vals)
            dynamics = dynamics / torch.abs(max_eig_scale)

        input_data = model.encoder(data).T
        preds = torch.zeros(pred_steps, input_data.shape[0], input_data.shape[1])
        for i in range(pred_steps):
            # current_pred = model.linear_dynamics(input_data)
            current_pred = torch.matmul(dynamics, input_data)
            preds[i] = current_pred
            input_data = current_pred

        dyn_fft = torch.zeros_like(torch.abs(torch.fft.rfft(preds[:, 0, 0])))
        for i in range(0, preds.shape[1], fft_stride):
            for j in range(0, preds.shape[2], fft_stride):
                fft = torch.abs(torch.fft.rfft(preds[:, i, j]))
                fft[fft == torch.nan] == 0.0
                dyn_fft += fft

        dyn_fft /= torch.sum(dyn_fft[1:])
        # dyn_fft /= torch.max(dyn_fft[1:])

    return dyn_fft


def recon_psd(data, model, pred_steps, fft_stride=1):
    input_data = data
    preds = torch.zeros(pred_steps, input_data.shape[0], input_data.shape[1])
    for i in range(pred_steps):
        current_pred = model.predict(input_data)
        preds[i] = current_pred
        input_data = current_pred

    dyn_fft = torch.zeros_like(torch.abs(torch.fft.rfft(preds[:, 0, 0])))
    for i in range(0, preds.shape[1], fft_stride):
        for j in range(0, preds.shape[2], fft_stride):
            fft = torch.abs(torch.fft.rfft(preds[:, i, j]))
            fft[fft == torch.nan] == 0.0
            dyn_fft += fft

    dyn_fft /= torch.sum(dyn_fft[1:])

    return dyn_fft


# # Define the power law function
# def power_law(x, a, b):
#     return a * (x ** b)

# # Sample data
# x_data = np.array([1, 2, 3, 4, 5])
# y_data = np.array([1.2, 2.3, 3.5, 4.6, 5.7])

# # Fit the power law to the data
# params, covariance = curve_fit(power_law, x_data, y_data)

# # Extract the parameters
# a, b = params

# # Print the parameters
# print("a:", a)
# print("b:", b)
