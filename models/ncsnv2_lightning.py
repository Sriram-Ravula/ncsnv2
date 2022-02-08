import torch
from pytorch_lightning import LightningModule

from models import get_sigmas
from models.ncsnv2 import NCSNv2Deepest

from losses.losses_lightning import NCSN_Loss

class EMA(nn.Module):
    """ Model Exponential Moving Average V2 from timm"""
    def __init__(self, model, decay=0.999):
        super().__init__()
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
        self.score = NCSNv2Deepest(self.config)
        self.score_ema = EMA(self.score, self.config.model.ema_rate) if self.config.model.ema else self.score

        self.criterion = NCSN_Loss(self.config)

        #below this are tensors, so they need to be registered as module attributes in order to move automatically
        self.register_buffer("sigmas", get_sigmas(self.config))

        if self.config.data.dataset == 'RTM_N':
            n_shots = np.asarray(self.config.model.n_shots).squeeze()
            n_shots = torch.from_numpy(n_shots)
            #make sure it has dimension > 0 if it is a singleton (useful for indexing)
            if n_shots.numel() == 1:
                n_shots = torch.unsqueeze(n_shots, 0)
            self.register_buffer("n_shots", n_shots)
        else:
            self.n_shots = None

        self.log_sample_path = os.path.join(self.args.log_path, 'samples')
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

        if self.config.model.sigma_dist == 'rtm_dynamic':
            loss, sigmas_running, n_shots_count = self.criterion(model, batch, self.sigmas, n_shots=self.n_shots)
            out_dict = {'loss': loss,
                        'n_shots_count': n_shots_count,
                        'sigmas_running': sigmas_running}
        else:
            loss = self.criterion(model, batch, self.sigmas, n_shots=self.n_shots)
            out_dict = {'loss': loss}
        
        return out_dict
    
    def training_step(self, batch, batch_idx):
        train_dict = self.shared_step(batch, batch_idx, val=False)
        
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

            #update the score network's sigmas list
            self.score.set_sigmas(self.sigmas)
            self.score_ema.set_sigmas(self.sigmas)

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
    
    def on_validation_epoch_end(self):
        """Checks the need to sample and then samples if necessary"""
        if self.current_epoch % self.config.training.snapshot_freq == 0:
            self.sample_rtm()
        
    def anneal_langevin_rtm(self, batch):
        #we rteally need to introduce the dataloader to be able to function here!

    def anneal_langevin_dsm(self):
        num_samples = self.config.sampling.batch_size // self.trainer.world_size #number of samples to make on this GPU

        x_mod = torch.rand(num_samples, self.config.data.channels,
                                  self.config.data.image_size, self.config.data.image_size).type_as(self.sigmas)
        
        n_steps_each = self.config.sampling.n_steps_each
        step_lr = self.config.sampling.step_lr
        final_only = self.config.sampling.final_only
        verbose = self.args.verbose
        denoise = self.config.sampling.denoise

        images = []
        
        with torch.no_grad():
            for c, sigma in enumerate(self.sigmas):
                labels = torch.ones(x_mod.shape[0).type_as(self.sigmas) * c 
                labels = labels.long()

                step_size = step_lr * (sigma / self.sigmas[-1]) ** 2

                for s in range(n_steps_each):
                    grad = self(x_mod, labels)

                    noise = torch.randn_like(x_mod) 

                    #Langevin update step
                    x_mod = x_mod + step_size * grad + np.sqrt(step_size * 2) * noise 

                    


