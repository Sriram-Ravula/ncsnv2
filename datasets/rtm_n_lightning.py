from pytorch_lightning import LightningDataModule
import torchvision.transforms as transforms

class RTMDataModule(LightningDataModule):
    def __init__(self, path, config):
        self.path = path
        self.config = config

        self.transform = transforms.Compose([transforms.Resize(size = [config.data.image_size, config.data.image_size],
                                             interpolation=transforms.InterpolationMode.BICUBIC)])
        
        #holds the slice directory IDs
        self.slices = [] 

        #hold the true height and width of the images
        self.H = 0
        self.W = 0

    def prepare_data(self):
        #grb all the valid sids and set image dimensions
        for dI in os.listdir(self.path):

            slice_path = os.path.join(self.path, dI)

            if os.path.isdir(slice_path):

                img_path = os.path.join(slice_path, 'image.npy')
                shots_path = os.path.join(slice_path, 'shots')
                config_path = os.path.join(slice_path, 'config.yaml')
                velocity_path = os.path.join(slice_path, 'slice.npy')

                if os.path.isfile(img_path) and os.path.isdir(shots_path) and os.path.isfile(config_path) and os.path.isfile(velocity_path):

                    self.slices.append(dI)

                    if self.H == 0:

                        self.W, self.H = load_exp(slice_path)['vel'].shape
        
        self.tensors = self.__build_dataset__()
