import os
import random
import torch
from tensorboardX import SummaryWriter


class Logger(SummaryWriter):
    def __init__(self, logdir):
        super(Logger, self).__init__(logdir)
        self.model_name = ""
    
    def set_model_name(self, model_name):
        self.model_name = model_name

    def log(self, mode, epoch, loss, acc):
        self.add_scalar(f"{self.model_name}_{mode}_loss", loss, epoch)
        self.add_scalar(f"{self.model_name}_{mode}_accuracy", acc, epoch)
        
    def log_lr(self, epoch, learning_rate):
        self.add_scalar(f"{self.model_name}_learning_rate", learning_rate, epoch)


class Printer:
    def __init__(self, log_dir=''):
        self.log_dir = os.path.join(log_dir, "log.txt")

    def set_log_dir(self, log_dir):
        self.log_dir = os.path.join(log_dir, "log.txt")

    def __call__(self, line, end="\n"):
        print(line, end=end)
        with open(self.log_dir, 'a') as f:
            f.write(f"{line}{end}")