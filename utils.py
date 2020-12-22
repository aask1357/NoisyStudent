import torch
import torch.optim as optim
import copy
from models import load_model
import ast

def update_params(config, params):
    for param in params:
        k, v = param.split("=")
        k_split = k.split('.')
        
        if len(k_split) > 1:
            k_parent = k_split[0]
            k_child = '.'.join(k_split[1:])
            cur_param = [f"{k_child}={v}"]
            update_params(config[k_parent], cur_param)
        elif k in config:
            try:
                v = ast.literal_eval(v)
            except:
                pass
            config[k] = v
        else:
            print(f"{k}, {v} params not updated")


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']            
    

class LR_Lambda:
    def __init__(self, lr_decay_rate, lr_decay_epoch):
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_epoch = lr_decay_epoch
        self.epoch = lr_decay_epoch
    
    def __call__(self, epoch):
        if epoch >= self.epoch:
            self.epoch += self.lr_decay_epoch
            return self.lr_decay_rate
        else:
            return 1.0
        

class Loader:
    def __init__(self, teacher_path, student_path, models, epochs, ratio,
                 learning_rate,lr_decay_rate, lr_decay_epoch, device, num_classes,
                 dropout_prob, stochastic_depth_prob):
        self.teacher_path = teacher_path
        self.student_path = student_path
        self.models = models
        self.epochs = epochs
        self.ratio = ratio
        self.learning_rate = learning_rate
        self.lr_decay_rate = lr_decay_rate
        self.lr_decay_epoch = lr_decay_epoch
        self.device = device
        self.num_classes = num_classes
        self.dropout_prob = dropout_prob
        self.stochastic_depth_prob = stochastic_depth_prob
        
    def load(self, phase):
        if self.teacher_path != "":
            checkpoint = torch.load(self.teacher_path)
            phase = checkpoint["phase"]
            teacher = load_model(self.models[phase], num_classes=self.num_classes,
                                 dropout_prob=self.dropout_prob,
                                 stochastic_depth_prob=self.stochastic_depth_prob)
            phase += 1
            teacher.load_state_dict(checkpoint['best_model'])
            teacher.to(self.device)
            self.teacher_path = ""  # only at the first time
            print(f"teacher loaded. Phase: {phase}")
        else:
            teacher = None
        
        if self.student_path != "":
            checkpoint = torch.load(self.student_path)
            phase = checkpoint["phase"]
            model = load_model(self.models[phase], num_classes=self.num_classes,
                                 dropout_prob=self.dropout_prob,
                                 stochastic_depth_prob=self.stochastic_depth_prob)
            model.load_state_dict(checkpoint['model'])
            model.to(self.device)
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate[phase], momentum=0.9)
            optimizer.load_state_dict(checkpoint['optimizer'])
            
            lr_lambda = LR_Lambda(self.lr_decay_rate[phase], self.lr_decay_epoch[phase])
            scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)
            scheduler.load_state_dict(checkpoint['scheduler'])
            start_epoch = checkpoint['epoch']
            self.student_path = ""  # only at the first time
            
            best_acc = checkpoint['best_acc']
            best_epoch = checkpoint['best_epoch']
            best_model = checkpoint['best_model']
            print(f"student loaded.")
        else:
            model = load_model(self.models[phase], num_classes=self.num_classes,
                               dropout_prob=self.dropout_prob,
                               stochastic_depth_prob=self.stochastic_depth_prob).to(self.device)
            optimizer = optim.SGD(model.parameters(), lr=self.learning_rate[phase], momentum=0.9)
            lr_lambda = LR_Lambda(self.lr_decay_rate[phase], self.lr_decay_epoch[phase])
            scheduler = optim.lr_scheduler.MultiplicativeLR(optimizer, lr_lambda)
            start_epoch, iteration = 0, 0
            
            best_acc = 0.
            best_epoch = 0
            best_model = copy.deepcopy(model.state_dict())
        
        epoch = self.epochs[phase]
        
        return teacher, model, optimizer, scheduler, start_epoch, epoch, phase, best_acc, best_epoch, best_model
