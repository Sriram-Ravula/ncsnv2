import torch
from datasets.rtm_n import RTM_N

def anneal_dsm_score_estimation(scorenet, samples, sigmas, labels=None, anneal_power=2., hook=None):
    if labels is None:
        labels = torch.randint(0, len(sigmas), (samples.shape[0],), device=samples.device)
    used_sigmas = sigmas[labels].view(samples.shape[0], *([1] * len(samples.shape[1:]))) #grab the correct sigma_i value
    noise = torch.randn_like(samples) * used_sigmas
    perturbed_samples = samples + noise #x~ = x + N(0, sigma_i)
    target = - 1 / (used_sigmas ** 2) * noise # -(x~ - x)/(sigma_i)
    scores = scorenet(perturbed_samples, labels) #s_theta(x~, sigma_i)
    target = target.view(target.shape[0], -1)
    scores = scores.view(scores.shape[0], -1)
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** anneal_power

    if hook is not None:
        hook(loss, labels)

    return loss.mean(dim=0)
"""
Args:
    scorenet: s_{theta} the score-based network
    samples: (X, y) pairs where y is for identifying the RTM image index! [N, C, H, W]
    n_shots: the list of n_shots_i that we are using (e.g. [1, 2, 5, 10, 50, 100]). [nshots]
    lambdas_list: the list of lambda(n_shots_i) values to scale the loss by. [nshots]
    dynamic_lambdas: whether or not we want to calclate lambda as the MSE between rtm_243 and rtm_n during runtime
    labels: the index of the n_shots we are using for each pixel - i.e. i in n_shots_i. [N].
    hook: a hook for experiment logging
"""
def rtm_score_estimation(scorenet, samples, n_shots, lambdas_list, rtm_dataset, dynamic_lambdas=False, labels=None, hook=None):
    #(1) if we aren't given an index i for n_shots, pick a random one
    #labels has size [N] - same as samples[0]
    if labels is None: 
        labels = torch.randint(0, len(n_shots), (samples[0].shape[0],), device=samples[0].device)
    
    n_shots = torch.tensor(n_shots)
    lambdas_list = torch.tensor(lambdas_list)

    #(2) grab the n_shots_i (e.g. 180 shots)
    #used_nshots has size [N]
    used_nshots = n_shots[labels].view(samples[0].shape[0], *([1] * len(samples[0].shape[1:]))) 

    #(3) grab the correct lambda(n_shots_i) val
    #lambda_n has size [N]
    lambda_n = lambdas_list[labels].view(samples[0].shape[0], *([1] * len(samples[0].shape[1:])))

    #(4) grab the n_shots images corresponding to the rtm243 images we have as training samples
    #we pass the RTM_243 image with its index and the n_shots_i to the function and get back the RTM_{n_shots_i} image
    #preturbed_samples = [N, C, H, W]
    perturbed_samples = rtm_dataset.dataset.grab_rtm_image(samples, used_nshots) #x_{n_shots_i}

    #(5) form the targets 
    #targets = [N, C, H, W]
    target = samples[0] - perturbed_samples #(x_243 - x_{n_shots_i})

    if dynamic_lambdas:
        lambda_n =  torch.sqrt(torch.mean(target ** 2, dim=[1, 2, 3])) #this gives us sqrt((1 / H*W*C) ||x_243 - x_N||_2^2), i.e. the RMSE per sample

        #now we want to keep a running average of the lambda values for each of the n_shot_i values
        #we want a list of length [nshots] with each entry having the sum of all MSEs in this iteration corresponding to that n_shot_i
        #we also want a list of length [nshots] where we count the number of times we encounter each n_shot_i value

        sum_mses_list = torch.zeros(len(n_shots)).float().to(samples[0].device)
        n_shots_count = torch.zeros(len(n_shots)).float().to(samples[0].device)

        for i, shot_idx in enumerate(labels.cpu().numpy().squeeze()):
            sum_mses_list[shot_idx] = sum_mses_list[shot_idx] + lambda_n[i].item()
            n_shots_count[shot_idx] = n_shots_count[shot_idx] + 1  

        lambda_n = lambda_n.view(samples[0].shape[0], *([1] * len(samples[0].shape[1:])))                       

    target = target / (lambda_n ** 2) #TODO this is an addition to try out
    target = target.view(target.shape[0], -1)

    #(6) grab the network output
    #scores = [N, C, H, W]
    if dynamic_lambdas:
        scores = scorenet.forward_with_sigmas(perturbed_samples, lambda_n)
    else:
        scores = scorenet(perturbed_samples, labels) #s_theta(x~, lambda_i) / lambda(n_shots_i)
    scores = scores.view(scores.shape[0], -1)

    #(7) calculate the loss
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * lambda_n.squeeze() ** 2 #TODO added the power factor to lambda

    if hook is not None:
        hook(loss, labels)

    if dynamic_lambdas:
        return loss.mean(dim=0), sum_mses_list, n_shots_count 
    else:
        return loss.mean(dim=0)
