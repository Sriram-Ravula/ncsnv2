import torch
import torchvision
from torchvision.utils import make_grid, save_image
from pytorch_lightning import LightningModule

from models import get_sigmas, anneal_Langevin_dynamics
from models.ncsnv2 import NCSNv2Deepest

from losses.losses_lightning import NCSN_Loss

class EMA(nn.Module):
    """
    Model Exponential Moving Average V2 from timm.
    EMA that expects to be updates after every step of the main model.
    Updates have the form W = decay * W + (1 - decay) * W. 
    """
    def __init__(self, model, decay=0.999):
        """
        Args:
            model: The model to do EMA on.
                   Type: nn.Module.
            decay: The EMA rate.
                   Type: float. 
                   Default value: 0.999.  
        """
        super().__init__()
        self.module = torch.copy.deepcopy(model)
        self.module.eval()
        self.decay = decay
    
    def forward(self, x, y, sigmas=None):
        """
        Performs a forward pass using the underlying EMA model. 
        """
        return self.module(x, y, sigmas)

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

        self.log_sample_path = os.path.join(self.args.log_path, 'samples')
        os.makedirs(self.log_sample_path, exist_ok=True)

        #these are nn.modules so they automatically move between devices
        self.score = NCSNv2Deepest(self.config)
        self.score_ema = EMA(self.score, self.config.model.ema_rate) if self.config.model.ema else self.score

        self.criterion = NCSN_Loss(self.config)

        #below this are tensors, so they need to be registered as module attributes in order to move automatically
        self.register_buffer("sigmas", get_sigmas(self.config))

        if self.config.data.dataset == 'RTM_N':
            n_shots = np.asarray(self.config.model.n_shots).squeeze()
            n_shots = torch.from_numpy(n_shots)
            if n_shots.numel() == 1:
                n_shots = torch.unsqueeze(n_shots, 0)
            self.register_buffer("n_shots", n_shots)
        else:
            self.register_buffer("n_shots", None)
        
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
    
    def forward(self, x, noise_level_idcs, noise_levels=None):
        """Returns the output from the EMA model.
           If not using EMA, returns output from score network."""
        return self.score_ema(x, noise_level_idcs, noise_levels)
    
    def shared_step(self, batch, batch_idx, val=False):
        if val:
            model = self.score_ema
        else:
            model = self.score

        if self.config.model.sigma_dist == 'rtm_dynamic':
            loss, sigmas_running, n_shots_count = self.criterion(model, batch, self.sigmas, n_shots=self.n_shots, val=val)
            out_dict = {'loss': loss,
                        'n_shots_count': n_shots_count,
                        'sigmas_running': sigmas_running}
        else:
            loss = self.criterion(model, batch, self.sigmas, n_shots=self.n_shots, val=val)
            out_dict = {'loss': loss}
        
        return out_dict
    
    def training_step(self, batch, batch_idx):
        if batch_idx == 0 and self.current_epoch == 0:
            grid = make_grid(batch[0], nrow=4, normalize=True)
            self.logger.experiment.add_image('RTM243_train_sample', grid, self.current_epoch)
            if self.config.data.dataset == 'RTM_N':
                grid = make_grid(batch[2], nrow=4, normalize=True)
                self.logger.experiment.add_image('RTMN_train_sample', grid, self.current_epoch)

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
            self.sigmas = self.current_epoch * self.sigmas + (sigmas_running / total_n_shots_count)
            self.sigmas /= (self.current_epoch + 1)

            #update the score network's sigmas list
            self.score.set_sigmas(self.sigmas)
            if self.config.model.ema:
                self.score_ema.module.set_sigmas(self.sigmas)

            #log the current values
            if self.trainer.is_global_zero:
                np.save(os.path.join(self.log_sample_path, 'sigmas_{}.npy'.format(self.current_epoch)), self.sigmas.cpu().numpy())

                self.log("sigmas", self.sigmas, prog_bar=False,  logger=True,  rank_zero_only=True)
                self.log("n_shots_count", self.n_shots_count, prog_bar=False, logger=True, rank_zero_only=True)
                self.log("sigmas_running", self.sigmas_running, prog_bar=False, logger=True, rank_zero_only=True)

                self.logger.experiment.add_histogram(tag="sigmas", values=self.sigmas, global_step=self.trainer.global_step)
                self.logger.experiment.add_histogram(tag="n_shots_count", values=self.n_shots_count, global_step=self.trainer.global_step)
                self.logger.experiment.add_histogram(tag="sigmas_running", values=self.sigmas_running, global_step=self.trainer.global_step)

    def on_validation_epoch_start(self):
        """Checks if it is appropriate to sample"""
        if self.current_epoch % self.config.training.snapshot_freq == 0:
            if self.config.data.dataset == 'RTM_N':
                batch = next(iter(self.val_dataloader()))
            else:
                batch = None

            samples = self.sample(batch)
    
    def validation_step(self, batch, batch_idx):
        """Calculates test loss on a batch"""
        if batch_idx == 0 and self.current_epoch == 0:
            grid = make_grid(batch[0], nrow=4, normalize=True)
            self.logger.experiment.add_image('RTM243_val_sample', grid, self.current_epoch)
            if self.config.data.dataset == 'RTM_N':
                grid = make_grid(batch[2], nrow=4, normalize=True)
                self.logger.experiment.add_image('RTMN_val_sample', grid, self.current_epoch)

        val_dict = self.shared_step(batch, batch_idx, val=True)

        self.log("val_loss", val_dict['loss'], prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True)
    
    def sample(self, batch):
        num_samples = self.config.sampling.batch_size // self.trainer.world_size #number of samples to make on this GPU

        if self.config.data.dataset == 'RTM_N':
            init_samples = batch[2] #the RTM_N image
            rtm_243 = batch[0]
            if init_samples.shape[0] > num_samples:
                init_samples = init_samples[:num_samples]
                rtm_243 = rtm_243[:num_samples]

            image_grid = make_grid(init_samples, nrow=4, normalize=True)
            save_image(image_grid,
                        os.path.join(self.args.log_sample_path, 'init_epoch{}_{}.png'.format(self.current_epoch, self.global_rank)))
            self.logger.experiment.add_image('init_sample', image_grid, self.current_epoch)

            image_grid = make_grid(rtm_243, nrow=4, normalize=True)
            save_image(image_grid,
                        os.path.join(self.args.log_sample_path, 'rtm243__epoch{}_{}.png'.format(self.current_epoch, self.global_rank)))
            self.logger.experiment.add_image('rtm243_sample', image_grid, self.current_epoch)

        else:
            init_samples = torch.rand(num_samples, self.config.data.channels,
                                        self.config.data.image_size, self.config.data.image_size).type_as(batch[0])
        
        out_samples = anneal_Langevin_dynamics(x_mod=init_samples, 
                                               scorenet=self.score_ema, 
                                               sigmas=self.sigmas, 
                                               n_steps_each=self.config.sampling.n_steps_each, 
                                               step_lr=self.config.sampling.step_lr,
                                               final_only=self.config.sampling.final_only, 
                                               verbose=True, 
                                               denoise=self.config.sampling.denoise, 
                                               add_noise=(self.config.data.dataset != 'RTM_N'))
        
        image_grid = make_grid(out_samples, nrow=4, normalize=True)
        save_image(image_grid,
                    os.path.join(self.args.log_sample_path, 'samples_epoch{}_{}.png'.format(self.current_epoch, self.global_rank)))
        self.logger.experiment.add_image('output_sample', image_grid, self.current_epoch)

        mse = torch.nn.MSELoss(recuction='mean')
        self.log("sample_loss", mse(out_samples, rtm_243), prog_bar=True, logger=True)
        
        return out_samples
                    