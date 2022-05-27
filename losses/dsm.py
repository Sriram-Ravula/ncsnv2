import torch
from datasets.rtm_n import RTM_N

def anneal_dsm_score_estimation(scorenet, batch, sigmas, labels=None, anneal_power=2., hook=None):
    samples, _ = batch
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

    return loss.mean(dim=0) #loss.sum() #

def supervised_conditional(scorenet, batch, n_shots, sigmas, dynamic_sigmas=False, anneal_power=2., val=False):
    """
    Expects batch to consist of (RTM243, slice_id, RTM_N, N)
    """
    X_243, slice_id, X_N, shot_idx = batch
    
    #(2) grab the correct sigma(shot_idx) val
    #sigma_n has size [N]
    with torch.no_grad():
        if dynamic_sigmas:
            #this gives us sqrt((1 / H*W*C) ||x_243 - x_N||_2^2), i.e. the RMSE per sample
            sigma_n = torch.sqrt(torch.mean((X_243 - X_N) ** 2, dim=[1, 2, 3])) 

            sum_mses_list = torch.zeros(n_shots.numel(), device=X_243.device)
            n_shots_count = torch.zeros(n_shots.numel(), device=X_243.device)

            for i, idx in enumerate(shot_idx):
                sum_mses_list[idx] += sigma_n[i].item()
                n_shots_count[idx] += 1

            sigma_n = sigma_n.view(X_243.shape[0], *([1] * len(X_243.shape[1:])))                 
        else:
            sigma_n = sigmas[shot_idx].view(X_243.shape[0], *([1] * len(X_243.shape[1:])))
    
    #(3) Scale the target
    target = X_243.view(X_243.shape[0], -1)

    #(4) grab the network output s_theta(x~, sigma_i) / sigma(n_shots_i)
    #labels has size [N] (batch size)
    #scores = [N, C, H, W]
    labels = torch.tensor(shot_idx, device=X_243.device).long()
    if dynamic_sigmas and not val:
        #if dynamic, pass the measured sigma values to the score network to scale it
        scores = scorenet(X_N, labels, sigma_n)
    else:
        scores = scorenet(X_N, labels, None) 
    scores = scores.view(scores.shape[0], -1)
    
    #(5) calculate the loss
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * sigma_n.squeeze() ** anneal_power

    if dynamic_sigmas:
        return loss.mean(dim=0), sum_mses_list, n_shots_count 
        #return loss.sum(), sum_mses_list, n_shots_count
    else:
        loss.mean(dim=0) #return loss.sum() #

def supervised_loss(scorenet, batch, anneal_power=2):
    """
    Calculates a supervised l2 loss and sums over losses for values of k, each scaled by empirical mse
    """
    X_243, slice_id, X_N, shot_idx = batch

    #(1) find the empirical RMSE for each sample
    #sigma_n has size [N]
    with torch.no_grad():
        #this gives us sqrt((1 / H*W*C) ||x_243 - x_N||_2^2), i.e. the RMSE per sample
        sigma_n = torch.sqrt(torch.mean((X_243 - X_N) ** 2, dim=[1, 2, 3])) 

        sigma_n = sigma_n.view(X_243.shape[0], *([1] * len(X_243.shape[1:])))

    #(2) Get score network output
    labels = torch.tensor(shot_idx, device=X_243.device).long()
    scores = scorenet(X_N, labels)
    scores = scores.view(scores.shape[0], -1)

    #(3) make and shape the target
    target = X_243.view(X_243.shape[0], -1)

    #(4) grab the loss
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * sigma_n.squeeze() ** anneal_power

    return loss.mean(dim=0)


def rtm_loss(scorenet, batch, n_shots, sigmas, dynamic_sigmas=False, anneal_power=2., val=False):
    """
    Expects batch to consist of (RTM243, slice_id, RTM_N, N)
    """
    X_243, slice_id, X_N, shot_idx = batch
    
    #(1) form the targets 
    #targets has siz [N, C, H, W]
    target = X_243 - X_N #(x_243 - x_{n_shots_i})
    
    #(2) grab the correct sigma(shot_idx) val
    #sigma_n has size [N]
    with torch.no_grad():
        if dynamic_sigmas:
            #this gives us sqrt((1 / H*W*C) ||x_243 - x_N||_2^2), i.e. the RMSE per sample
            sigma_n = torch.sqrt(torch.mean(target ** 2, dim=[1, 2, 3])) 

            sum_mses_list = torch.zeros(n_shots.numel(), device=X_243.device)
            n_shots_count = torch.zeros(n_shots.numel(), device=X_243.device)

            for i, idx in enumerate(shot_idx):
                sum_mses_list[idx] += sigma_n[i].item()
                n_shots_count[idx] += 1

            sigma_n = sigma_n.view(X_243.shape[0], *([1] * len(X_243.shape[1:])))                 
        else:
            sigma_n = sigmas[shot_idx].view(X_243.shape[0], *([1] * len(X_243.shape[1:])))
    
    #(3) Scale the target
    target = target / (sigma_n ** 2)
    target = target.view(target.shape[0], -1)

    #(4) grab the network output s_theta(x~, sigma_i) / sigma(n_shots_i)
    #labels has size [N] (batch size)
    #scores = [N, C, H, W]
    labels = torch.tensor(shot_idx, device=X_243.device).long()
    if dynamic_sigmas and not val:
        #if dynamic, pass the measured sigma values to the score network to scale it
        scores = scorenet(X_N, labels, sigma_n)
    else:
        scores = scorenet(X_N, labels, None) 
    scores = scores.view(scores.shape[0], -1)
    
    #(5) calculate the loss
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * sigma_n.squeeze() ** anneal_power

    if dynamic_sigmas:
        return loss.mean(dim=0), sum_mses_list, n_shots_count 
        #return loss.sum(), sum_mses_list, n_shots_count
    else:
        loss.mean(dim=0) #return loss.sum() #

def rtm_score_estimation(scorenet, samples, n_shots, lambdas_list, rtm_dataset, dynamic_lambdas=False, labels=None, hook=None, val=False):
    """
    Args:
        scorenet: s_{theta} the score-based network
        samples: (X, y, flipped) pair of rtm_243 image, index, and whether flipped. (tensor:[N, 1, H, W], list:[N], list:[N])
        n_shots: The list of n_shots_i that we are using (e.g. [1, 2, 5, 10, 50, 100]). Torch tensor [nshots].
        lambdas_list: The list of lambda(n_shots_i) values to scale the loss by. Torch Tensor [nshots].
        rtm_dataset: The dataset to use when gathering the RTM_n images corresponding to the input.
        dynamic_lambdas: Whether or not we want to calclate lambda as the RMSE between rtm_243 and rtm_n during runtime.
        labels: The index of the n_shots we are using for each pixel - i.e. i in n_shots_i. Torch tensor [N].
        hook: A hook for experiment logging.
    """
    
    #(1) if we aren't given an index i for n_shots, pick a random one
    #labels has size [N] - same as samples[0]
    if labels is None: 
        labels = torch.randint(0, n_shots.numel(), (samples[0].shape[0],), device=samples[0].device)

    #(2) grab the n_shots_i (e.g. 180 shots)
    #used_nshots has size [N]
    used_nshots = n_shots[labels].view(samples[0].shape[0], *([1] * len(samples[0].shape[1:]))) 

    #(3) grab the correct lambda(n_shots_i) val (this value will be replaced with empirical value if dynamic_lambdas=True)
    #lambda_n has size [N]
    lambda_n = lambdas_list[labels].view(samples[0].shape[0], *([1] * len(samples[0].shape[1:])))

    #(4) grab the n_shots images corresponding to the rtm243 images we have as training samples
    #we pass the RTM_243 image with its index and the n_shots_i to the function and get back the RTM_{n_shots_i} image
    #preturbed_samples has size [N, C, H, W]
    perturbed_samples = rtm_dataset.dataset.grab_rtm_image(samples, used_nshots).type_as(samples[0]) #x_{n_shots_i}

    #(5) form the targets 
    #targets has siz [N, C, H, W]
    target = samples[0] - perturbed_samples #(x_243 - x_{n_shots_i})

    if dynamic_lambdas:
        #this gives us sqrt((1 / H*W*C) ||x_243 - x_N||_2^2), i.e. the RMSE per sample
        lambda_n =  torch.sqrt(torch.mean(target ** 2, dim=[1, 2, 3])) 

        #now we want to keep a running average of the lambda values for each of the n_shot_i values
        #we want a list of length [nshots] with each entry having the sum of all MSEs in this iteration corresponding to that n_shot_i
        #we also want a list of length [nshots] where we count the number of times we encounter each n_shot_i value
        sum_mses_list = torch.zeros(n_shots.numel()).float().to(samples[0].device)
        n_shots_count = torch.zeros(n_shots.numel()).float().to(samples[0].device)

        for i, shot_idx in enumerate(labels.cpu().numpy().squeeze()):
            sum_mses_list[shot_idx] += lambda_n[i].item()
            n_shots_count[shot_idx] += 1  

        lambda_n = lambda_n.view(samples[0].shape[0], *([1] * len(samples[0].shape[1:])))                       

    target = target / (lambda_n ** 2) #TODO this is an addition to try out
    target = target.view(target.shape[0], -1)

    #(6) grab the network output s_theta(x~, lambda_i) / lambda(n_shots_i)
    #scores = [N, C, H, W]
    if dynamic_lambdas and not val:
        #if dynamic, pass the measured lambda values to the score network to scale it
        scores = scorenet(perturbed_samples, labels, lambda_n)
    else:
        scores = scorenet(perturbed_samples, labels, None) 
    scores = scores.view(scores.shape[0], -1)

    #(7) calculate the loss
    loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * lambda_n.squeeze() ** 2 #TODO added the power factor to lambda

    if hook is not None:
        hook(loss, labels)

    if dynamic_lambdas:
        return loss.mean(dim=0), sum_mses_list, n_shots_count 
    else:
        return loss.mean(dim=0)
