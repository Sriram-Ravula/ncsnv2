import torch
from pytorch_lightning import LightningModule

from models import get_sigmas
from models.ncsnv2 import NCSNv2Deepest
from models.ema import EMAHelper

from losses.dsm import anneal_dsm_score_estimation, rtm_score_estimation

class NCSNv2_Lightning(LightningModule):
    def __init__(self, args, config):
        super().__init__()
        
        self.config = config
        self.args = args

        self.score = NCSNv2Deepest(config)
        #self.test_score = self.score #used in testing

        #TODO ema introduces a lot of distributed- and GPU-rank related questions
        #Test this thoroughly
        if self.config.model.ema:
            #(I think) this will be instantiated on only the rank-0 GPU of each node since it is not nn.module
            self.ema_helper = EMAHelper(mu=self.config.model.ema_rate) 
            self.ema_helper.register(self.score, False)
            #Will be used during validation. We want it on the appropriate GPU so we will initialize 
            #self.test_score = self.ema_helper.ema_copy(self.score, False)   

        self.sigmas = get_sigmas(self.config).to(self.device) #TODO do we need a .to() for a tensor or does it know where to go?

        if self.config.data.dataset == 'RTM_N':
            self.n_shots = np.asarray(self.config.model.n_shots).squeeze()
            self.n_shots = torch.from_numpy(self.n_shots).type_as(self.sigmas)
            #make sure it has dimension > 0 if it is a singleton (useful for indexing)
            if self.n_shots.numel() == 1:
                self.n_shots = torch.unsqueeze(self.n_shots, 0)
            
            #If we are dyamically altering lambdas, start a count of each n_shot encountered in training and a running sum of MSEs for each n_shot
            if self.config.model.sigma_dist == 'rtm_dynamic':
                def mean_reduce(x):
                    """Helper function for reducing a tensor only along its batch dimension"""
                    return torch.mean(x, dim=0) 
                self.reduction_fn = mean_reduce
                self.total_n_shots_count = torch.zeros(self.n_shots.numel()).type_as(self.sigmas)
                self.sigmas_running = get_sigmas(self.config).type_as(self.sigmas)

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
        return self.score(x, noise_level_idcs)
    
    def training_step(self, batch, batch_idx):
        X, y = batch #assume the transform is absorbed into the datamodule

        if self.config.model.sigma_dist == 'rtm':
            loss = rtm_score_estimation(self.score, (X, y), self.n_shots, self.sigmas, dataset, 
                                        dynamic_lambdas=False, labels=None)
            train_dict = {'loss': loss}
        elif self.config.model.sigma_dist == 'rtm_dynamic':
            loss, sum_mses_list, n_shots_count = rtm_score_estimation(self.score, (X, y), self.n_shots, self.sigmas, dataset, 
                                                    dynamic_lambdas=True, labels=None)
            train_dict = {'loss': loss,
                          'n_shots_count': n_shots_count,
                          'sum_mses_list': sum_mses_list}
        else:
            loss = anneal_dsm_score_estimation(self.score, X, self.sigmas, None, self.config.training.anneal_power)
            train_dict = {'loss': loss}

        self.log("train_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True, reduce_fx='mean')

        return train_dict
    
    def on_train_batch_end(self, outputs):
        """Update the EMA weights. Done at the end of the training batch after backwards and step.
           Reduce the shot count and running sigmas across all GPUs and calculate updated sigma""""
        if self.config.model.ema:
            self.ema_helper.update(self.score)
        
        if self.config.model.sigma_dist != 'rtm_dynamic':
            return
        
        #update the running shot count and sigmas across all devices
        shot_counts = outputs['n_shots_count']
        sigmas_list = outputs['sum_mses_list']

        for _, count in shot_counts.items():
            self.total_n_shots_count += n_shots_count 
        for _, sigmas in sigmas_list.items():
            self.sigmas_running += sigmas_list
        
        #calculate the new value of sigma 
        self.sigmas = self.sigmas_running / self.total_n_shots_count #update the master sigmas list
        
        #log the current values
        self.log("sigmas", self.sigmas, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True, reduce_fx=self.reduction_fn)
        self.log("n_shots_count", self.n_shots_count, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True, reduce_fx=self.reduction_fn)
        self.log("sigmas_running", self.sum_mses_lists, prog_bar=False, on_step=True, on_epoch=True, logger=True, sync_dist=True, reduce_fx=self.reduction_fn)

        self.logger.experiment.add_histogram(tag="sigmas", values=self.sigmas, global_step=self.trainer.global_step)
        self.logger.experiment.add_histogram(tag="n_shots_count", values=self.n_shots_count, global_step=self.trainer.global_step)
        self.logger.experiment.add_histogram(tag="sigmas_running", values=self.sum_mses_lists, global_step=self.trainer.global_step)
        
    
    def on_validation_start(self):
        """Sets up the test model"""
        if self.config.model.ema:
            self.test_score = self.ema_helper.ema_copy(self.score, False).to(self.device)
        else:
            self.test_score = self.score
        self.test_score.eval()

    def validation_step(self, batch, batch_idx):
        """Calculates test loss on a batch"""
        X, y = batch

        if self.config.model.sigma_dist == 'rtm':
            loss = rtm_score_estimation(self.test_score, (X, y), self.n_shots, self.sigmas, dataset, 
                                        dynamic_lambdas=False, labels=None)
        elif self.config.model.sigma_dist == 'rtm_dynamic':
            loss, _, _ = rtm_score_estimation(self.test_score, (X, y), self.n_shots, self.sigmas, dataset, 
                                                    dynamic_lambdas=True, labels=None)
        else:
            loss = anneal_dsm_score_estimation(self.test_score, X, self.sigmas, None, self.config.training.anneal_power)

        self.log("val_loss", loss.item(), prog_bar=True, on_step=True, on_epoch=True, logger=True, sync_dist=True, reduce_fx='mean')

        if self.current_epoch % self.config.training.snapshot_freq == 0:
            self.sample_rtm()
    
    def on_validation_epoch_end(self):
        """Checks the need to sample and then samples if necessary"""
        if self.config.model.sigma_dist == 'rtm_dynamic':
            self.score.set_sigmas(self.sigmas)
            if self.global_rank == 0:
                np.save(os.path.join(self.log_sample_path, 'lambdas_{}.npy'.format(self.trainer.global_step)), self.sigmas.cpu().numpy())
        
    def sample_rtm(self):
        #we rteally need to introduce the dataloader to be able to function here!


    
    def on_validation_end(self):
        """Tears down the test model"""
        del self.test_score
        self.test_score = None

