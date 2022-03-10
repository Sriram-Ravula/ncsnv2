# NCSNv2 For K-Shot RTM Images

## Running Experiments

### Relevant Files
- `main_ddp.py`: used to launch and run training sessions
- `ddp_runner.py`: main program for training with DDP
- `dsm.py`: contains denoising score matching and k-shot RTM losses
- `datasets/`
  - `__init__.py`: prepares the datasets and transforms
  - `rtm_n.py`: Seam rtm k-shot dataset
  - `velocity_fine.py`: velocity or full-shot RTM dataset (loads from preprocessed and saved torch tensor)
  - `ibalt.py`: Ibalt k-shot rtm dataset
- `configs/`
  - `ddp_test_1.yml`: config for a short training run of Seam RTM k-shot dataset
  - `ddp_test_2.yml`: short run of 243-shot Seam RTM (using regular NCSN training with denoising score matching)
  - `ddp_test_3.yml`: config for a short training run of Ibalt RTM k-shot dataset 

### Dependencies

Both Conda- and Pip-friendly requirements files are provided. You may have to strip the repo and version names from the files before trying to install the packages to make it play nice. 

### Project structure

`main_ddp.py` is the file that you should run for both training and sampling. Execute ```python main_ddp.py --help``` to get its usage description:

```
usage: main_ddp.py [-h] [--config CONFIG] [--seed SEED] [--exp EXP] [--doc DOC]
                   [--comment COMMENT] [--verbose VERBOSE] [--resume_training] 
                   [--ni]

optional arguments:
  -h, --help            show this help message and exit
  --config CONFIG       Path to the config file 
  --seed SEED           Random seed
  --exp EXP             Path for saving running related data.
  --doc DOC             A string for documentation purpose. Will be the name
                        of the log folder.
  --comment COMMENT     A string for experiment comment
  --verbose VERBOSE     Verbose level: info | debug | warning | critical
  --resume_training     Whether to resume training
  -i IMAGE_FOLDER, --image_folder IMAGE_FOLDER
                        The folder name of samples
  --ni                  No interaction. Suitable for Slurm Job launcher
```

Configuration files are in `config/`. You don't need to include the prefix `config/` when specifying  `--config` . All files generated when running the code is under the directory specified by `--exp`. They are structured as:

```bash
<exp> # a folder named by the argument `--exp` given to main.py
├── datasets # all dataset files (for our case, can contain precomputed full-shot RTM images) 
├── logs # contains checkpoints and samples produced during training
│   └── <doc> # a folder named by the argument `--doc` specified to main.py
│      ├── checkpoint_x.pth # the checkpoint file saved at the x-th training iteration
│      ├── config.yml # the configuration file for training this model
│      ├── stdout.txt # all outputs to the console during training
│      └── samples # all samples produced during training
└── tensorboard # tensorboard files for monitoring training
    └── <doc> # this is the log_dir of tensorboard
```

### Training

For example, we can train an NCSNv2 on Seam RTM k-shot using:

```bash
python3 main_ddp.py --config ddp_rtmn.yml --seed 42 --exp /scratch/04703/sravula/experiments --doc seam_rtm --verbose debug --ni
```

Log files will be saved in `<exp>/logs/seam_rtm`.
