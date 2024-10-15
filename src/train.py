import data
import models
from utils import parameter, util, Trainer

args = util.init_config(parameter.args)

util.init_env(args)

datasets = data.build_dataset(args)

model = models.build_model(args)

loss, optimizer = models.build_loss_and_optimizer(args, model)

trainer = Trainer.Trainer(args, datasets=datasets, model=model,criterion=loss,optimizer=optimizer)

trainer.begin()
