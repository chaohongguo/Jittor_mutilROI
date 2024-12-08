import os
import sys
import time
import jittor
from tqdm import tqdm
from tensorboardX import SummaryWriter


class BaseTrainer(object):
    """
    Base class for Trainer objects.
    Takes care of checkpointing/logging/resuming training
    """

    def __init__(self, options):
        self.options = options
        self.endtime = time.time() + self.options.time_to_run

        # override this function to define your specific feature
        # such as train_ds,model,optimizer
        self.models_dict = None
        self.optimizers_dict = None
        self.checkpoint = None
        self.train_ds = None
        self.eval_ds = None
        self.model = None
        self.optimizer = None
        self.smpl = None
        self.criterion = None
        self.bbox_type = None
        self.crop_w = None
        self.crop_h = None
        self.bbox_size = None
        self.joints_idx = None
        self.joints_num = None
        self.n_views = None
        self.init_fn()
        self.summary_writer = SummaryWriter(self.options.summary_dir)
        from utils import CheckpointSaver
        self.saver = CheckpointSaver(save_dir=options.checkpoint_dir)

        if self.options.resume and self.saver.exists_checkpoint():
            self.checkpoint = self.saver.load_checkpoint(self.models_dict, self.optimizers_dict,
                                                         checkpoint_file=None)

        if self.checkpoint is None:
            self.epoch_count = 0
            self.step_count = 0
        else:
            self.epoch_count = self.checkpoint['epoch']
            self.step_count = self.checkpoint['total_step_count']

    def train(self):
        # Iterate over all epochs
        for epoch in tqdm(range(self.epoch_count, self.options.num_epochs),
                          total=self.options.num_epochs, initial=self.epoch_count):
            from utils import CheckpointDataLoader
            train_data_loader = CheckpointDataLoader(self.train_ds, checkpoint=self.checkpoint,
                                                     batch_size=self.options.batch_size,
                                                     num_workers=self.options.num_workers,
                                                     shuffle=self.options.shuffle_train)
            # Iterate over all batches in an epoch
            for step, batch in enumerate(tqdm(train_data_loader, desc='Epoch ' + str(epoch),
                                              total=len(self.train_ds) // self.options.batch_size,
                                              initial=train_data_loader.checkpoint_batch_idx),
                                         train_data_loader.checkpoint_batch_idx):
                if time.time() < self.endtime:
                    batch = {k: v if isinstance(v, jittor.Var) else v for k, v in batch.items()}
                    out = self.train_step(batch)
                    self.step_count += 1
                    if self.step_count % self.options.summary_steps == 0:
                        self.train_summaries(batch, *out)
                    # if self.step_count % self.options.checkpoint_steps == 0:
                    #     self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step,
                    #                                self.options.batch_size, self.step_count)
                    #     tqdm.write('STEP Checkpoint saved')

                    if (step+1) % self.options.test_steps == 0:
                        eval_result = self.test()
                        # TODO more logs record
                        with open(os.path.join(self.options.log_dir, 'test_log.txt'), mode='a',
                                  encoding='utf-8') as logger:
                            print('Epoch {}-Step{}: '.format(epoch + 1, step + 1), eval_result, file=logger)
                        self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, step,
                                                   self.options.batch_size, self.step_count, eval_result=eval_result)

                else:
                    tqdm.write('Timeout reached')
                    # self.finalize()
                    self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch, 0,
                                               self.options.batch_size)
                    tqdm.write('Checkpoint saved')
                    sys.exit(0)
            # load a checkpoint only on startup, for the next epochs
            # just iterate over the dataset as usual
            self.checkpoint = None

            # if (epoch + 1) % self.options.save_epochs == 0:
            #     self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch + 1, 0,
            #                                self.options.batch_size, self.step_count, )
            #     tqdm.write('Checkpoint saved')

            # if (epoch + 1) % self.options.test_epochs == 0 and (epoch + 1) >= self.options.start_test_epoch:
            if (epoch + 1) % self.options.test_epochs == 0:
                eval_result = self.test()
                # TODO more logs record
                with open(os.path.join(self.options.log_dir, 'test_log.txt'), mode='a', encoding='utf-8') as logger:
                    print('Epoch {}-Step{}: '.format(epoch + 1, step), eval_result, file=logger)
                self.saver.save_checkpoint(self.models_dict, self.optimizers_dict, epoch + 1, 0,
                                           self.options.batch_size, self.step_count, eval_result=eval_result)
        return

    def load_pretrained(self, checkpoint_file=None):
        if checkpoint_file is not None:
            checkpoint = jittor.load(checkpoint_file)
            for model in self.models_dict:
                if model in checkpoint:
                    self.models_dict[model].load_state_dict(checkpoint[model], strict=False)
                    print('Checkpoint loaded')

    # The following methods (with the possible exception of test) have to be implemented in the derived classes
    def init_fn(self):
        raise NotImplementedError('You need to provide an _init_fn method')

    def train_step(self, input_batch):
        raise NotImplementedError('You need to provide a _train_step method')

    def train_summaries(self, input_batch):
        raise NotImplementedError('You need to provide a _train_summaries method')

    def test(self):
        raise NotImplementedError('You need to provide a _test method')
