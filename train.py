from utils import TrainOptions
from train import Trainer_mutilROI

if __name__ == '__main__':
    options = TrainOptions().parse_args()
    print(f'==================using model {options.model_name}=====================')
    if options.model_name == 'mutilROI':
        trainer = Trainer_mutilROI(options)
    trainer.train()
