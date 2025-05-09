import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import DataLoader
from fcdiffusion.dataset import TrainDataset
from fcdiffusion.logger import ImageLogger
from fcdiffusion.model import create_model, load_state_dict
import torch
torch.cuda.set_device(0)


# Configs
# resume_path = r'D:\paper\FCDiffusion_code-main\lightning_logs\fcdiffusion_mid_pass_checkpoint\epoch=11-step=241999.ckpt'
resume_path = r'D:\paper\FRE_FCD\lightning_logs_SA\fcdiffusion_high_pass_checkpoint\epoch=7-step=9999.ckpt'
batch_size = 2
logger_freq = 1000
learning_rate = 1e-5 #origin is 6e-6,1e-5
sd_locked = True
val_every_n_train_steps = 2000

# First use cpu to load configs. Pytorch Lightning will automatically move it to GPUs.
model = create_model('configs/model_config.yaml').cpu()
model.load_state_dict(load_state_dict(resume_path, location='cpu'))
model.learning_rate = learning_rate
model.sd_locked = sd_locked
control_mode = model.control_mode
logger_root_path = 'fcdiffusion_' + control_mode + '_img_logs'
checkpoint_path = 'fcdiffusion_' + control_mode + '_checkpoint'

# with open("model_arch.txt",'w') as f:
#     f.write(model)

dataset = TrainDataset('datasets/training_data.json',cache_size=1000)
print("dataset ok")
dataloader = DataLoader(dataset, num_workers=0, batch_size=batch_size, shuffle=True)
print("dataloader ok")
logger = ImageLogger(root_path=logger_root_path, batch_frequency=logger_freq)
val_checkpoint = ModelCheckpoint(dirpath='lightning_logs_SA/' + checkpoint_path,
                                 every_n_train_steps=val_every_n_train_steps, save_top_k=-1)
trainer = pl.Trainer(gpus='0', precision=32, callbacks=[logger, val_checkpoint])

# Train!
trainer.fit(model, dataloader)
