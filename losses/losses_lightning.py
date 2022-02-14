import torch

class NCSN_Loss(torch.nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config

        self.loss_type = 'rtm' if "rtm" in self.config.model.sigma_dist else "dsm"
        self.dynamic_sigmas = True if self.config.model.sigma_dist == 'rtm_dynamic' else False
        self.anneal_power = self.config.training.anneal_power
    
    def forward(self, scorenet, batch, sigmas, n_shots=None, labels=None, val=False):
        if self.loss_type == 'rtm':
            return self.rtm_loss(scorenet, batch, n_shots, sigmas, val)
        else:
            return self.dsm_loss(scorenet, batch, sigmas, labels)
    
    def dsm_loss(self, scorenet, batch, sigmas, labels=None):
        """
        expects batch consist of (RTM243, slice_id)
        """
        X, _ = batch 
        if labels is None:
            labels = torch.randint(0, len(sigmas), (X.shape[0],)).type_as(X)
        used_sigmas = sigmas[labels].view(X.shape[0], *([1] * len(X.shape[1:]))) #grab the correct sigma_i value
        noise = torch.randn_like(X) * used_sigmas
        perturbed_samples = X + noise #x~ = x + N(0, sigma_i)
        target = - 1 / (used_sigmas ** 2) * noise # -(x~ - x)/(sigma_i)
        scores = scorenet(perturbed_samples, labels) #s_theta(x~, sigma_i)
        target = target.view(target.shape[0], -1)
        scores = scores.view(scores.shape[0], -1)
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * used_sigmas.squeeze() ** self.anneal_power

        return loss.mean(dim=0)
    
    def rtm_loss(self, scorenet, batch, n_shots, sigmas, val=False):
        """
        Expects batch to consist of (RTM243, slice_id, RTM_N, N)
        """
        X_243, slice_id, X_N, shot_idx = batch
        
        #(1) form the targets 
        #targets has siz [N, C, H, W]
        target = X_243 - X_N #(x_243 - x_{n_shots_i})
        
        #(2) grab the correct sigma(shot_idx) val
        #sigma_n has size [N]
        if self.dynamic_sigmas:
            #this gives us sqrt((1 / H*W*C) ||x_243 - x_N||_2^2), i.e. the RMSE per sample
            sigma_n =  torch.sqrt(torch.mean(target ** 2, dim=[1, 2, 3])) 

            sum_mses_list = torch.zeros(n_shots.numel()).type_as(X_243)
            n_shots_count = torch.zeros(n_shots.numel()).type_as(X_243)

            for i, idx in enumerate(shot_idx):
                sum_mses_list[idx] = sum_mses_list[idx] + sigma_n[i].item()
                n_shots_count[idx] += 1

            sigma_n = sigma_n.view(X_243.shape[0], *([1] * len(X_243.shape[1:])))                 
        else:
            sigma_n = sigmas[labels].view(X_243.shape[0], *([1] * len(X_243.shape[1:])))
        
        #(3) Scale the target
        target = target / (sigma_n ** 2)
        target = target.view(target.shape[0], -1)

        #(4) grab the network output s_theta(x~, sigma_i) / sigma(n_shots_i)
        #labels has size [N] (batch size)
        #scores = [N, C, H, W]
        labels = torch.tensor(shot_idx).type_as(X_243)
        if self.dynamic_sigmas and not val:
            #if dynamic, pass the measured sigma values to the score network to scale it
            scores = scorenet(X_N, labels, sigma_n)
        else:
            scores = scorenet(X_N, labels, None) 
        scores = scores.view(scores.shape[0], -1)
        
        #(5) calculate the loss
        loss = 1 / 2. * ((scores - target) ** 2).sum(dim=-1) * sigma_n.squeeze() ** self.anneal_power

        if self.dynamic_lambdas:
            return loss.mean(dim=0), sum_mses_list, n_shots_count 
        else:
            return loss.mean(dim=0)











        

        

