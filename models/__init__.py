import torch
import numpy as np

def get_sigmas(config):
    if config.model.sigma_dist == 'geometric':
        sigmas = torch.tensor(
            np.exp(np.linspace(np.log(config.model.sigma_begin), np.log(config.model.sigma_end),
                               config.model.num_classes))).float().to(config.device)
    elif config.model.sigma_dist == 'uniform':
        sigmas = torch.tensor(
            np.linspace(config.model.sigma_begin, config.model.sigma_end, config.model.num_classes)
        ).float().to(config.device)

    else:
        raise NotImplementedError('sigma distribution not supported')

    return sigmas

@torch.no_grad()
def anneal_Langevin_dynamics(x_mod, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True, add_noise=True):
    images = []

    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c #dummy target 1...T depending on iteration
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2
            for s in range(n_steps_each):
                grad = scorenet(x_mod, labels)

                #choose whether to add random noise during each gradient ascent step
                if add_noise:
                    noise = torch.randn_like(x_mod) 
                else:
                    noise = torch.zeros_like(x_mod)

                #calculate l2 norms of gradient (score) and the additive noise for logging
                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()

                x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2) #core Langevin step

                #calc l2 norm of iterate variable for logging
                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()

                #calc snr as scaled version of [||s(x, \sigma_i)|| / ||z_t||] and mean of score for logging
                snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm
                grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    print("level: {}, step_size: {}, grad_norm: {}, image_norm: {}, snr: {}, grad_mean_norm: {}".format(
                        c, step_size, grad_norm.item(), image_norm.item(), snr.item(), grad_mean_norm.item()))

        #final denoising step if desired - removes the very last additive z_L 
        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images

@torch.no_grad()
def langevin_Inverse(x_mod, y, A, scorenet, sigmas, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False, denoise=True, add_noise=True, 
                             decimate_sigma=None, mode=None, true_x=None):
    images = []

    #if desired, decimate the number of noise scales to speed up inference
    if decimate_sigma is not None:
        sigmas_temp = sigmas[0:-1:decimate_sigma].tolist() #grab every decimate_sigma'th value except the last one
        sigmas_temp.append(sigmas[-1]) #add the last sigma value back to the list
        # num_sigmas = sigmas.shape[0] // decimate_sigma
        # sigmas_temp = []
        # for i in range(num_sigmas):
        #    sigmas_temp.append(sigmas[-1])
        sigmas = sigmas_temp #swap the new decimated sigma list for the main one

    mse = torch.nn.MSELoss()

    N, C, H, W = x_mod.shape

    steps = np.geomspace(start=5, stop=1, num=len(sigmas))

    c2 = 1

    with torch.no_grad():
        #outer loop over noise scales
        for c, sigma in enumerate(sigmas):
            #dummy target 1...T depending on iteration
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c 
            labels = labels.long()

            #step_size = step_lr * (sigma / sigmas[-1]) ** 2
            step_size = steps[c]

            #Inner loop over T
            for s in range(n_steps_each):
                #s(x_t) ~= \grad_x log p(x) -- THE PRIOR
                grad = scorenet(x_mod, labels)

                prior_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                #prior_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2 * sigma ** 2

                #calculate the maximum likelihood gradient - i.e. MSE gradient
                #A should be [N, m, C * H * W], x should be [N, C, H, W], y should be [N, m, 1]
                if mode=='denoising':
                    Axt = x_mod 
                    mle_grad = (Axt - y) * (1 / N) #for denoising, y has same dimension as x
                else:
                    Axt = torch.matmul(A, x_mod.view(N, -1, 1))
                    mle_grad = torch.matmul(torch.transpose(A, -2, -1), Axt - y).view(N, C, H, W) * c2 #MSE gradient
                    #mle_grad = torch.matmul(torch.transpose(A, -2, -1), torch.sign(Axt - y)).view(N, C, H, W) * (1 / N) #L1 error gradient

                likelihood_norm = torch.norm(mle_grad.view(mle_grad.shape[0], -1), dim=-1).mean()
                #likelihood_mean_norm = torch.norm(mle_grad.mean(dim=0).view(-1)) ** 2

                if c == 0 and s == 0:
                    c2 = prior_norm.item() / likelihood_norm.item()
                    mle_grad = mle_grad * c2 #MSE gradient
                    likelihood_norm = torch.norm(mle_grad.view(mle_grad.shape[0], -1), dim=-1).mean()


                #The final gradient
                grad = grad - mle_grad

                grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
                #grad_mean_norm = torch.norm(grad.mean(dim=0).view(-1)) ** 2

                #choose whether to add random noise during each gradient ascent step
                if add_noise:
                    noise = torch.randn_like(x_mod) 
                else:
                    noise = torch.zeros_like(x_mod)

                x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2) #core Langevin step

                #calc l2 norm of iterate variable for logging
                image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
                noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
                snr = np.sqrt(step_size / 2.) * prior_norm / noise_norm
                mse_iter = mse(Axt, y)
                if true_x is not None:
                    mse_true = mse(true_x, x_mod)

                if not final_only:
                    images.append(x_mod.to('cpu'))
                if verbose:
                    print("\nlevel: {}, step_size: {:.4f}, prior_norm: {:.4f}, likelihood_norm: {:.4f}, grad_norm: {:.4f} \
                            image_norm: {:.4f}, train_mse: {:.4f}".format( \
                        c, step_size, prior_norm.item(), likelihood_norm.item(), grad_norm.item(), image_norm.item(), \
                        mse_iter.item()))
                    
                    if true_x is not None:
                        print("true_mse: {:.4f}".format(mse_true.item()))

        #final denoising step if desired - removes the very last additive z_L 
        if denoise:
            last_noise = (len(sigmas) - 1) * torch.ones(x_mod.shape[0], device=x_mod.device)
            last_noise = last_noise.long()
            x_mod = x_mod + sigmas[-1] ** 2 * scorenet(x_mod, last_noise)
            images.append(x_mod.to('cpu'))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images

@torch.no_grad()
def inverse_solver(x_mod, y, A, scorenet, sigmas, lr = [5, 1], c1=1, c2=1, auto_c2=True,
                   final_only=False, verbose=False, likelihood_every=1,
                   decimate_sigma=None, mode=None, true_x=None, sigma_type = 'subsample', likelihood_type="l2"):
    images = []

    #if desired, decimate the number of noise scales to speed up inference
    if decimate_sigma is not None:
        if sigma_type == 'subsample': #grab equally-spaced sigma values
            sigmas_temp = sigmas[0:-1:decimate_sigma].tolist() 
            sigmas_temp.append(sigmas[-1]) 

        elif sigma_type == 'last': #grab just the last sigma value multiple times
            num_sigmas = sigmas.shape[0] // decimate_sigma
            sigmas_temp = []
            for i in range(num_sigmas):
                sigmas_temp.append(sigmas[-1])

        else:
            sigmas_temp = sigmas

        sigmas = sigmas_temp 

    mse = torch.nn.MSELoss()

    N, C, H, W = x_mod.shape

    steps = np.geomspace(start=lr[0], stop=lr[1], num=len(sigmas))

    likelihood_norm = 0

    with torch.no_grad():
        if sigma_type == 'last':
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * 1099 
            labels = labels.long()
        for c, sigma in enumerate(sigmas):
            if sigma_type == 'subsample':
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * decimate_sigma * c
                labels = labels.long()
            elif sigma_type != 'last':
                labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
                labels = labels.long()

            step_size = steps[c]

            #s(x_t) ~= \grad_x log p(x) -- THE PRIOR
            grad = scorenet(x_mod, labels) * c1

            prior_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()

            if c % likelihood_every == 0:
                #\grad_x log p(y | x) -- LIKELIHOOD
                if mode=='denoising':
                    Axt = x_mod
                    if likelihood_type == "l2":
                        mle_grad = (Axt - y) * c2 
                    elif likelihood_type == "l1":
                        mle_grad = torch.sign(Axt - y) * c2 
                else:
                    Axt = torch.matmul(A, x_mod.view(N, -1, 1)) 
                    if likelihood_type == "l2":
                        mle_grad = torch.matmul(torch.transpose(A, -2, -1), Axt - y).view(N, C, H, W) * c2 
                    elif likelihood_type == "l1":
                        mle_grad = torch.matmul(torch.transpose(A, -2, -1), torch.sign(Axt - y)).view(N, C, H, W) * c2 

                likelihood_norm = torch.norm(mle_grad.view(mle_grad.shape[0], -1), dim=-1).mean()

                if auto_c2 and c == 0:
                    c2 = prior_norm.item() / likelihood_norm.item()
                    mle_grad = mle_grad * c2 #MSE gradient
                    likelihood_norm = torch.norm(mle_grad.view(mle_grad.shape[0], -1), dim=-1).mean()

                grad = grad - mle_grad

            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()

            x_mod = x_mod + step_size * grad
            #x_mod = torch.clamp(x_mod, 0.0, 1.0)

            #calc l2 norm of iterate variable for logging
            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()
            mse_iter = mse(Axt, y)
            if true_x is not None:
                mse_true = mse(true_x, x_mod)

            if not final_only:
                images.append(x_mod.cpu())
            if verbose:
                print("\n iteration: {}, sigma: {:.4f}, step_size: {:.4f}, prior_norm: {:.4f}, likelihood_norm: {:.4f}, grad_norm: {:.4f} \
                        image_norm: {:.4f}, train_mse: {:.4f}".format( \
                    c, sigma, step_size, prior_norm.item(), likelihood_norm.item(), grad_norm.item(), image_norm.item(), \
                    mse_iter.item()))
                
                if true_x is not None:
                    print("true_mse: {:.4f}".format(mse_true.item()))

        if final_only:
            return [x_mod.to('cpu')]
        else:
            return images

@torch.no_grad()
def anneal_Langevin_dynamics_inpainting(x_mod, refer_image, scorenet, sigmas, image_size,
                                        n_steps_each=100, step_lr=0.000008):
    """
    Currently only good for 32x32 images. Assuming the right half is missing.
    """

    images = []

    #refer_image is the untainted x (?)
    #right now this only works with 3-channel images
    refer_image = refer_image.unsqueeze(1).expand(-1, x_mod.shape[1], -1, -1, -1)
    refer_image = refer_image.contiguous().view(-1, 3, image_size, image_size)

    
    x_mod = x_mod.view(-1, 3, image_size, image_size)
    cols = image_size // 2
    half_refer_image = refer_image[..., :cols]
    with torch.no_grad():
        for c, sigma in enumerate(sigmas):
            labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
            labels = labels.long()
            step_size = step_lr * (sigma / sigmas[-1]) ** 2

            for s in range(n_steps_each):
                images.append(x_mod.to('cpu'))
                corrupted_half_image = half_refer_image + torch.randn_like(half_refer_image) * sigma
                x_mod[:, :, :, :cols] = corrupted_half_image
                noise = torch.randn_like(x_mod) * np.sqrt(step_size * 2)
                grad = scorenet(x_mod, labels)
                x_mod = x_mod + step_size * grad + noise
                print("class: {}, step_size: {}, mean {}, max {}".format(c, step_size, grad.abs().mean(),
                                                                         grad.abs().max()))

        return images

@torch.no_grad()
def anneal_Langevin_dynamics_interpolation(x_mod, scorenet, sigmas, n_interpolations, n_steps_each=200, step_lr=0.000008,
                             final_only=False, verbose=False):
    images = []

    n_rows = x_mod.shape[0]

    x_mod = x_mod[:, None, ...].repeat(1, n_interpolations, 1, 1, 1)
    x_mod = x_mod.reshape(-1, *x_mod.shape[2:])

    for c, sigma in enumerate(sigmas):
        labels = torch.ones(x_mod.shape[0], device=x_mod.device) * c
        labels = labels.long()
        step_size = step_lr * (sigma / sigmas[-1]) ** 2
        for s in range(n_steps_each):
            grad = scorenet(x_mod, labels)

            noise_p = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            noise_q = torch.randn(n_rows, x_mod.shape[1], x_mod.shape[2], x_mod.shape[3],
                                  device=x_mod.device)
            angles = torch.linspace(0, np.pi / 2., n_interpolations, device=x_mod.device)

            noise = noise_p[:, None, ...] * torch.cos(angles)[None, :, None, None, None] + \
                        noise_q[:, None, ...] * torch.sin(angles)[None, :, None, None, None]

            noise = noise.reshape(-1, *noise.shape[2:])
            grad_norm = torch.norm(grad.view(grad.shape[0], -1), dim=-1).mean()
            noise_norm = torch.norm(noise.view(noise.shape[0], -1), dim=-1).mean()
            image_norm = torch.norm(x_mod.view(x_mod.shape[0], -1), dim=-1).mean()

            x_mod = x_mod + step_size * grad + noise * np.sqrt(step_size * 2)

            snr = np.sqrt(step_size / 2.) * grad_norm / noise_norm

            if not final_only:
                images.append(x_mod.to('cpu'))
            if verbose:
                print(
                    "level: {}, step_size: {}, image_norm: {}, grad_norm: {}, snr: {}".format(
                        c, step_size, image_norm.item(), grad_norm.item(), snr.item()))


    if final_only:
        return [x_mod.to('cpu')]
    else:
        return images