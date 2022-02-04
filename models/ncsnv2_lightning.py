import torch
from pytorch_lightning import LightningModule

from models import get_sigmas
from models.ncsnv2 import NCSNv2Deepest

from losses.dsm import anneal_dsm_score_estimation, rtm_score_estimation

class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, decay=0.999):
        super(EMA, self).__init__()
        # make a copy of the model for accumulating moving average of weights
        self.module = torch.copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
    
    def forward(self, x, y):
        return self.module(x, y)

    def _update(self, model, update_fn):
        """Performs EMA"""
        with torch.no_grad():
            for ema_v, model_v in zip(self.module.state_dict().values(), model.state_dict().values()):
                ema_v.copy_(update_fn(ema_v, model_v))

    def update(self, model):
        """Updates the EMA with the given model's weights"""
        self._update(model, update_fn=lambda e, m: self.decay * e + (1. - self.decay) * m)

    def set(self, model):
        """Used to set the EMA weights equal to model"""
        self._update(model, update_fn=lambda e, m: m)

class NCSNv2_Lightning(LightningModule):
    def __init__(self, args, config):
        super().__init__()
        
        self.config = config
        self.args = args

        #these are nn.modules so they automatically move between devices and stuff
        self.score = NCSNv2Deepest(config)
        self.score_ema = EMA(self.score, self.config.model.ema_rate) if self.config.model.ema else self.score

        #below this are tensors, so they need to be registered as module attributes in order to move automatically
        self.register_buffer("sigmas", get_sigmas(self.config))

        if self.config.data.dataset == 'RTM_N':
            n_shots = np.asarray(self.config.model.n_shots).squeeze()
            n_shots = torch.from_numpy(n_shots)
            #make sure it has dimension > 0 if it is a singleton (useful for indexing)
            if n_shots.numel() == 1:
                n_shots = torch.unsqueeze(n_shots, 0)
            self.register_buffer("n_shots", n_shots)
            
            #If we are dyamically altering lambdas, start a count of each n_shot encountered in training and a running sum of MSEs for each n_shot
            if self.config.model.sigma_dist == 'rtm_dynamic':
                def mean_reduce(x):
                    """Helper function for reducing a tensor only along its batch dimension"""
                    return torch.mean(x, dim=0) 
                self.reduction_fn = mean_reduce

        self.log_sample_path = os.path.join(args.log_path, 'samples')
        os.makedirs(self.log_sample_path, exist_ok=True)
        
    def configure_optimizers(self):
        lr = self.config.optim.lr
        weight_decay = self.config.optim.weight_decay
        beta1 = self.config.optim.beta1
        amsgrad = self.config.optim.amsgrad
        eps = self.config.optim.eps

        optim_type = self.config.optim.optimizer

        if optim_type == 'Adam':
            opt = torch.optim.Adam(self.score.parameters(), lr=lr, weight_decay=weight_decay,
                                    betas=(beta1, 0.999), amsgrad=amsgrad, eps=eps)
        elif optim_type == 'RMSProp':
            opt = torch.optim.RMSprop(self.score.parameters(), lr=lr, weight_decay=weight_decay)
        elif optim_type == 'SGD':
            opt =  torch.optim.SGD(self.score.parameters(), lr=lr, momentum=0.9)
        else:
            raise NotImplementedError('Optimizer {} not understood.'.format(config.optim.optimizer))
        
        return opt
    
    def forward(self, x, noise_level_idcs):
        """Returns the output from the EMA model.
           If not using EMA, returns output from score network."""
        return self.score_ema(x, noise_level_idcs)
    
    def shared_step(self, batch, batch_idx, val=False):
        if val:
            model = self.score_ema
        else:
            model = self.score

        X, y = batch #assume the transform is absorbed into the datamodule

        if self.config.model.sigma_dist == 'rtm':
            loss = rtm_score_estimation(model, (X, y), self.n_shots, self.sigmas, dataset, 
                                        dynamic_lambdas=False, labels=None)
            out_dict = {'loss': loss}
        elif self.config.model.sigma_dist == 'rtm_dynamic':
            loss, sigmas_running, n_shots_count = rtm_score_estimation(model, (X, y), self.n_shots, self.sigmas, dataset, 
                                                    dynamic_lambdas=True, labels=None)
            out_dict = {'loss': loss,
                        'n_shots_count': n_shots_count,
                        'sigmas_running': sigmas_running}
        else:
            loss = anneal_dsm_score_estimation(model, X, self.sigmas, None, self.config.training.anneal_power)
            out_dict = {'loss': loss}
        
        return out_dict
    
    def training_step(self, batch, batch_idx):
        train_dict = self.shared_step(batch, batch_idx)
        
        self.log("train_loss", train_dict['loss'], prog_bar=True, on_step=True, on_epoch=True, logger=True)

        return train_dict
    
    def on_train_batch_end(self, outputs):
        """Done at the end of the training batch after backwards and step
           Update the EMA weights.
            - must go here since we EMA expects to be updated after every score network update."""
        if self.config.model.ema:
            self.score_ema.update(self.score)
    
    def training_epoch_end(self, outputs):
        """Log important stuff for rtm_dynamic.
            - we do this here because we only want to log once an epoch to reduce overhead.
           Updates sigma stuff if we are doing rtm_dynamic.
            - we can put this here instead of at each training step end since score estimation with dynamic uses empirical sigmas""""
           
        if self.config.model.sigma_dist == 'rtm_dynamic':
            #TODO make sure we are iterating over the outputs correctly!
            #update the running shot count and sigmas across all devices
            total_n_shots_count = 0
            sigmas_running = 0

            for out in outputs
                shot_counts = out['n_shots_count']
                sigmas_list = out['sigmas_running']
                
                for count in shot_counts:
                    total_n_shots_count += count 
                for sigmas in sigmas_list:
                    sigmas_running += sigmas
            
            #calculate the new value of sigma 
            self.sigmas = self.current_epoch * self.sigmas + (self.sigmas_running / self.total_n_shots_count)
            self.sigmas /= (self.current_epoch + 1)

            #reset these guys so they don't overflow
            total_n_shots_count = 0
            sigmas_running = 0

            self.score.set_sigmas(self.sigmas)

            #log the current values
            if self.trainer.is_global_zero:
                np.save(os.path.join(self.log_sample_path, 'sigmas_{}.npy'.format(self.current_epoch)), self.sigmas.cpu().numpy())

                self.log("sigmas", self.sigmas, prog_bar=False,  logger=True,  rank_zero_only=True)
                self.log("n_shots_count", self.n_shots_count, prog_bar=False, logger=True, rank_zero_only=True)
                self.log("sigmas_running", self.sigmas_running, prog_bar=False, logger=True, rank_zero_only=True)

                self.logger.experiment.add_histogram(tag="sigmas", values=self.sigmas, global_step=self.trainer.global_step)
                self.logger.experiment.add_histogram(tag="n_shots_count", values=self.n_shots_count, global_step=self.trainer.global_step)
                self.logger.experiment.add_histogram(tag="sigmas_running", values=self.sigmas_running, global_step=self.trainer.global_step)

    def validation_step(self, batch, batch_idx):
        """Calculates test loss on a batch"""
        val_dict = self.shared_step(batch, batch_idx, val=True)

        self.log("val_loss", val_dict['loss'], prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)

        if self.current_epoch % self.config.training.snapshot_freq == 0:
            self.sample_rtm()
    
    def on_validation_epoch_end(self):
        """Checks the need to sample and then samples if necessary"""

        
    def sample_rtm(self):
        #we rteally need to introduce the dataloader to be able to function here!


