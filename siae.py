import torch
from torch import nn
import utils

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class DeepKoopmanAE(nn.Module):
    def __init__(
        self,
        input_size,
        output_size,
        layers=[256, 64],
        latent_dim=16,
        activ=nn.SELU(),
        manual_seed=1,
        trainable_dyn=True,
    ):
        super(DeepKoopmanAE, self).__init__()
        self.manual_seed = manual_seed
        torch.manual_seed(self.manual_seed)

        # Encoder
        encoder_layers = []
        if len(layers) == 0:
            encoder_layers.append(nn.Linear(input_size, latent_dim))
        else:
            encoder_layers.append(nn.Linear(input_size, layers[0]))
            encoder_layers.append(activ)
            for i in range(len(layers) - 1):
                encoder_layers.append(nn.Linear(layers[i], layers[i + 1]))
                encoder_layers.append(activ)
            encoder_layers.append(nn.Linear(layers[-1], latent_dim))
            encoder_layers.append(activ)
        self.encoder = nn.Sequential(*encoder_layers)

        # Linear dynamics layer
        self.linear_dynamics = nn.Linear(latent_dim, latent_dim)
        self.linear_dynamics.requires_grad_(trainable_dyn)

        # Decoder
        decoder_layers = []
        if len(layers) == 0:
            decoder_layers.append(nn.Linear(latent_dim, output_size))
        else:
            decoder_layers.append(nn.Linear(latent_dim, layers[-1]))
            decoder_layers.append(activ)
            for i in range(len(layers) - 1, 0, -1):
                decoder_layers.append(nn.Linear(layers[i], layers[i - 1]))
                decoder_layers.append(activ)
            decoder_layers.append(nn.Linear(layers[0], output_size))
            # decoder_layers.append(activ)
        self.decoder = nn.Sequential(*decoder_layers)

        self.init_all_weights()

    def init_all_weights(self):
        def init_weights(m):
            if isinstance(m, nn.Linear):
                torch.nn.init.xavier_uniform_(m.weight)
                # m.bias.data.fill_(0.01)

        self.encoder.apply(init_weights)
        self.linear_dynamics.apply(init_weights)
        self.decoder.apply(init_weights)

    def predict(self, xk):
        self.eval()
        with torch.no_grad():
            pred = self.forward(xk)
        return pred

    def reconstruct(self, xk):
        self.eval()
        with torch.no_grad():
            recon = self.recon_forward(xk)[0]
        return recon

    def forward(self, xk):
        """
        forward pass for the purpose of prediction
        """
        zk = self.encoder(xk)
        zk1 = self.linear_dynamics(zk)
        xk1_tilde = self.decoder(zk1)

        return xk1_tilde

    def detailed_forward(self, xk):
        """
        forward pass for the purpose of prediction
        """
        zk = self.encoder(xk)
        zk1 = self.linear_dynamics(zk)
        xk1_tilde = self.decoder(zk1)

        return xk1_tilde, zk1, zk

    def recon_forward(self, xk):
        """
        forward pass without dynamics - only reconstruction
        """
        zk = self.encoder(xk)
        xk_tilde = self.decoder(zk)
        return xk_tilde, zk

    def linearity_forward(self, xk):
        """
        just encode, wil this mess up back prop?
        """
        zk = self.encoder(xk)
        zk1 = self.linear_dynamics(zk)
        return zk1, zk

    def train_forward(self, x, y):
        """
        forward pass for training purposes
        """
        y_pred, zk1, zk = self.detailed_forward(x)
        xk_tilde, _ = self.recon_forward(x)
        y_encoded = self.encoder(y)
        return y_pred, zk1, zk, xk_tilde, y_encoded


class DummyReg(nn.Module):
    def __init__(self, f_scale=0.0, s_scale=0.0, f_exp=0.0, s_exp=0.0, schedule=None):
        super(DummyReg, self).__init__()
        self.f_scale = f_scale
        self.s_scale = s_scale
        self.f_exp = f_exp
        self.s_exp = s_exp
        self.schedule = schedule
        self.epoch = 0
        self.reg_hist = []

    def log(self, vals):
        pass

    def calc_losses(self, cont_eigen_vals):
        pass

    def inc_epoch(self):
        pass

    def forward(self, model, *args, **kwargs):
        return torch.sum(0 * model.linear_dynamics.weight)


class EigCoeffReg(nn.Module):
    def __init__(self, scale, pred_steps=10, power=-2, train_data=None, schedule=None):
        super(EigCoeffReg, self).__init__()
        self.train_data = train_data
        self.pred_steps = pred_steps
        self.power = power
        self.scale = scale
        self.schedule = schedule
        self.epoch = 0
        self.eig_coeff_norm = None
        if train_data is not None:
            self.target_psd = self.calc_target_psd(train_data)

    def calc_target_psd(self, data, fft_stride=1):
        with torch.no_grad():
            return utils.calc_target_psd(data, fft_stride=1)

    def representation_error(self, eig_fs, eig_coeffs, f, psd):
        representation_error = torch.zeros_like(f)
        for i, freq in enumerate(f):
            nearest_idx = torch.argmin(torch.abs(eig_fs - freq))
            # error = torch.abs(psd[i] - eig_coeffs[nearest_idx])**2 * torch.abs(freq - eig_fs[nearest_idx])**2
            error = self.target_psd[i] * torch.abs(freq - eig_fs[nearest_idx]) ** 2
            representation_error[i] = error
        total_representation_error = torch.mean(representation_error)
        return total_representation_error

    def representation_vals(self, eig_fs, eig_coeffs, f, psd):
        summed_powers = torch.zeros_like(eig_coeffs)

        # Loop through each frequency in f_tensor
        for i, freq in enumerate(f):
            # Find the index of the nearest eigenfrequency
            nearest_idx = torch.argmin(torch.abs(eig_fs - freq))

            # Add the power times the difference in frequencies to the corresponding eigenfrequency bin
            summed_powers[nearest_idx] += psd[i] ** 2 * torch.abs(
                freq - eig_fs[nearest_idx]
            )

        return summed_powers

    def interpolate_psd(self, eig_fs, f, psd):
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

    def calc_loss(self, model, X):
        dynamics = model.linear_dynamics.weight
        eigen_vals, eigen_vecs = torch.linalg.eig(dynamics)
        cont_eigen_vals = torch.log(eigen_vals)
        eig_fs = torch.abs(cont_eigen_vals.imag / torch.pi)

        with torch.no_grad():
            _, _, zk = model.detailed_forward(X)
        # TODO move mean to the next step, after cal eigen coeffs
        # avg_zk = torch.mean(zk, axis=0)
        avg_zk = zk
        eigen_coeffs = torch.abs(torch.matmul(avg_zk, torch.abs(eigen_vecs))) ** 2
        eigen_coeffs = torch.mean(eigen_coeffs, axis=0)
        # COMMENT
        # eigen_coeffs = eigen_coeffs * torch.exp(1*torch.abs(eigen_vals))

        if self.train_data is not None:
            f = torch.linspace(0, 1, len(self.target_psd))
            target = self.interpolate_psd(eig_fs, f, self.target_psd)
        else:
            target = torch.pow(1 + eig_fs, -2)

        if self.eig_coeff_norm is None:
            self.eig_coeff_norm = torch.sum(eigen_coeffs.detach())
        # eigen_coeffs = eigen_coeffs / torch.max(eigen_coeffs)
        # eigen_coeffs = eigen_coeffs / self.eig_coeff_norm
        eigen_coeffs = eigen_coeffs / sum(eigen_coeffs)
        target = target / torch.sum(target)
        # eigen_coeffs = eigen_coeffs / eigen_coeffs.shape[0]
        # target = target / self.target_psd.shape[0]
        error = target - eigen_coeffs
        error[eig_fs == 0] = 0

        if False:
            import matplotlib.pyplot as plt

            plt.xscale("log")
            plt.yscale("log")
            plt.plot(
                f.detach(),
                self.target_psd.detach() / target.detach().shape[0],
                c="dimgrey",
            )
            plt.scatter(eig_fs.detach(), eigen_coeffs.detach(), c="r")
            plt.scatter(eig_fs.detach(), target.detach(), c="k", marker="x")
            plt.show()

        reg_loss = torch.sum((error) ** 2) / error.shape[0]

        return reg_loss

    def forward(self, model, X, **kwargs):
        reg_loss = self.calc_loss(model, X)
        reg_loss = reg_loss * self.scale

        if self.schedule is not None:
            scale = self.schedule[self.epoch]
            reg_loss = reg_loss * scale
        return reg_loss

    def inc_epoch(self):
        self.epoch += 1


class SIAEEigCoeffReg(EigCoeffReg):
    def __init__(
        self, f_scale, f_exp, s_scale, s_exp, total_scale=1.0, *args, **kwargs
    ):
        super(SIAEEigCoeffReg, self).__init__(*args, **kwargs)
        self.total_scale = total_scale
        self.f_scale = f_scale
        self.f_exp = f_exp
        self.s_scale = s_scale
        self.s_exp = s_exp

    def calc_loss(self, model):
        dynamics = model.linear_dynamics.weight
        eigen_vals, eigen_vec = torch.linalg.eig(dynamics)
        cont_eigen_vals = torch.log(eigen_vals)

        masked_s_vals = torch.nn.functional.relu(cont_eigen_vals.real)
        s_loss = torch.mean(torch.abs(torch.pow(masked_s_vals, self.s_exp)))

        f_loss = torch.mean(torch.abs(torch.pow(cont_eigen_vals.imag, self.f_exp)))

        return f_loss, s_loss

    def forward(self, model, X, **kwargs):
        if self.schedule is not None:
            scale = self.schedule[self.epoch]
            if scale == 0.0:
                return torch.tensor(0.0)

        # only run long computation if actually needed
        if self.scale == 0.0:
            coeff_loss = 0.0
        else:
            coeff_loss = super().calc_loss(model, X)
            coeff_loss = coeff_loss * self.scale

        f_loss, s_loss = self.calc_loss(model)
        s_loss = s_loss * self.s_scale
        f_loss = f_loss * self.f_scale

        reg_loss = coeff_loss + f_loss + s_loss
        reg_loss = reg_loss * self.total_scale

        if self.schedule is not None:
            scale = self.schedule[self.epoch]
            reg_loss = reg_loss * scale
        return reg_loss


class TargetOpNorm(nn.Module):
    def __init__(self, target_op, scale, schedule=None, *args, **kwargs):
        super(TargetOpNorm, self).__init__()
        self.target_op = target_op
        self.scale = scale
        self.schedule = schedule
        self.epoch = 0
        # self.dmd_op = utils.compute_dmd_operator(train_data[:-1], train_data[1:])

    def log(self, vals):
        pass

    def inc_epoch(self):
        self.epoch += 1

    def forward(self, model, *args, **kwargs):
        if self.schedule is not None:
            scale = self.schedule[self.epoch]
            if scale == 0.0:
                return torch.tensor(0.0)

        reg_loss = torch.sum((model.linear_dynamics.weight - self.target_op) ** 2)

        if self.schedule is not None:
            scale = self.schedule[self.epoch]
            reg_loss = reg_loss * scale
        return reg_loss


class KoopmanLoss(nn.Module):
    def __init__(self, pred_scale=1.0, recon_scale=1.0, linearity_scale=0.1):
        super(KoopmanLoss, self).__init__()
        self.pred_scale = pred_scale
        self.recon_scale = recon_scale
        self.linearity_scale = linearity_scale

    def forward(self, *args, **kwargs):
        recon_loss, pred_loss, linearity_loss = self.koopman_loss_fn(*args, **kwargs)
        total_loss = (
            (self.pred_scale * pred_loss)
            + (self.recon_scale * recon_loss)
            + (self.linearity_scale * linearity_loss)
        )
        return total_loss, recon_loss, pred_loss, linearity_loss

    def koopman_loss_fn(self, x, zk, zk1, y_pred, xk_tilde, y_encoded, y):
        """
        Take in x -Encoder> zk -Dynamics> zk1 -Decoder> y_pred
        and compute losses
        needs to have weighting on the various losses?
        """
        recon_loss = torch.nn.functional.mse_loss(xk_tilde, x)
        pred_loss = torch.nn.functional.mse_loss(y_pred, y)
        linearity_loss = torch.nn.functional.mse_loss(zk1, y_encoded)
        # total_loss = recon_loss + pred_loss + linearity_loss
        return recon_loss, pred_loss, linearity_loss


def val(model, val_dataloader):
    model.eval()
    val_loss = 0.0

    with torch.no_grad():
        for x, y in val_dataloader:
            x, y = x.to(device), y.to(device)

            y_pred = model(x)
            val_loss += torch.nn.functional.mse_loss(y_pred, y).item()

    average_val_loss = val_loss / len(val_dataloader)
    return average_val_loss


def fit(
    model,
    train_dataloader,
    num_epochs,
    loss_obj,
    optimizer,
    regularizer=None,
    lr_scheduler=None,
    val_dataloader=None,
    model_logger=None,
):
    if lr_scheduler is None:
        lr_scheduler = torch.optim.lr_scheduler.LinearLR(
            optimizer, start_factor=1.0, end_factor=1.0
        )

    model.to(device)
    metrics = {
        "recon": [],
        "pred": [],
        "linearity": [],
        "reg": [],
        "loss": [],
        "val_loss": [],
    }

    for epoch in range(num_epochs):
        model.train()
        total_loss = 0.0
        total_koopman_losses = torch.zeros(3)

        for x, y in train_dataloader:
            x, y = x.to(device), y.to(device)

            optimizer.zero_grad()

            outputs = model.train_forward(x, y)
            y_pred, zk1, zk, xk_tilde, y_encoded = outputs

            loss, recon_loss, pred_loss, linearity_loss = loss_obj(
                x, zk, zk1, y_pred, xk_tilde, y_encoded, y
            )
            regularization_loss = regularizer(model, x)
            loss = loss + regularization_loss

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_koopman_losses += torch.tensor(
                [recon_loss, pred_loss, linearity_loss]
            )

        regularizer.inc_epoch()
        lr_scheduler.step()

        average_loss = total_loss / len(train_dataloader)
        avg_koopman_losses = total_koopman_losses / len(train_dataloader)

        if model_logger is not None:
            model_logger.log(model, epoch + 1)

        # Validation
        if val_dataloader is not None:
            average_val_loss = val(model, val_dataloader)
        else:
            average_val_loss = 0.0

        metrics["loss"].append(average_loss)
        metrics["recon"].append(avg_koopman_losses[0])
        metrics["pred"].append(avg_koopman_losses[1])
        metrics["linearity"].append(avg_koopman_losses[2])
        metrics["reg"].append(regularization_loss.detach())
        metrics["val_loss"].append(average_val_loss)

        print(
            f"Epoch [{epoch+1}/{num_epochs}], Loss: {average_loss:.5f}, Reg Loss: {regularization_loss.detach():.5f}, Val Loss: {average_val_loss:.5f}",
            end="\r",
        )
    print("")
    return metrics


if __name__ == "__main__":
    print("no main written for siae.py, check experiments")
