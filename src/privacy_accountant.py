import math
import torch
from scipy.special import logsumexp

def noise_injection(grad_tensor, sigma_g, max_norm, batch_size, use_cuda):
    """ Adds Gaussian noise to per-sample stochastic gradients.
    :param grad_tensor : stands for a stochastic gradient
    :param sigma_g : variance of the Gaussian noise (defined by DP theory)
    :param max_norm : clipping value
    :param batch_size : nb of data point in the considered batch"""
    std_gaussian = 2 * sigma_g * max_norm / batch_size
    # print(std_gaussian)
    gaussian_noise = torch.normal(0, std_gaussian, grad_tensor.shape)

    if use_cuda:
        gaussian_noise = gaussian_noise.to(torch.device('mps'))

    return grad_tensor + gaussian_noise

def compute_privacy_loss(trainer, client):
    p1 = compute_privacy_loss_basic(trainer, client)
    p2 = compute_privacy_loss_advance(trainer, client)
    return min(p1, p2)


def compute_privacy_loss_basic(trainer, client):
    return trainer.selected_times[client] * trainer.epsilon_setup[client] / (
            2 * trainer.q[client] * math.sqrt(trainer.clients_per_round * trainer.num_round))


def compute_privacy_loss_advance(trainer, client):
    return (2 * trainer.selected_times[client] * math.log(1 / trainer.delta[client]) ** 0.5 * trainer.epsilon_setup[
        client] / (2 * trainer.q[client] * math.sqrt(trainer.clients_per_round * trainer.num_round)) +
            trainer.selected_times[client] * trainer.epsilon_setup[client] / (
                    2 * trainer.q[client] * math.sqrt(trainer.clients_per_round * trainer.num_round)) * (
                    math.exp(trainer.epsilon_setup[client]) / (
                    2 * trainer.q[client] * math.sqrt(trainer.clients_per_round * trainer.num_round)) - 1))

def alpha_dp(trainer, client, alpha):
    log_moment_k = lambda k: math.lgamma(alpha + 1) - math.lgamma(k + 1) - math.lgamma(alpha - k + 1) \
                             + (alpha - k) * math.log(1 - trainer.s[client]) + k * math.log(trainer.s[client]) \
                             + (k ** 2 - k) / (2 * (trainer.sigma[client]/(2*trainer.w_clip/trainer.batch_size)) ** 2)
    all_log_moments_k = [log_moment_k(k) for k in range(0, alpha + 1)]
    return logsumexp(all_log_moments_k)

def compute_privacy_loss_MA(trainer, client):
    alpha_max = 32
    e = math.inf
    for alpha in range(1, alpha_max + 1):
        # print(alpha, alpha_dp(trainer, client, alpha))
        e = min((trainer.selected_times[client] * alpha_dp(trainer, client, alpha)
                 + math.log(1 / trainer.delta[client])) / alpha, e)
    return e
