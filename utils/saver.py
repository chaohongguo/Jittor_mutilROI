import os
import datetime
import jittor


class CheckpointSaver:
    def __init__(self, save_dir, save_steps=1000):
        self.save_dir = os.path.abspath(save_dir)
        self.save_steps = save_steps
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
        self._get_latest_checkpoint()
        return

    def exists_checkpoint(self, checkpoint_file=None):
        """
        check if checkpoint in current directory
        """
        if checkpoint_file is None:
            return False if self.latest_checkpoint is None else True
        else:
            return os.path.isfile(checkpoint_file)
        pass

    def save_checkpoint(self, models, optimizers, epoch, batch_idx, batch_size, total_step_count,
                        is_update=True, eval_result=None):
        """
        Save checkpoint.
        """
        timestamp = datetime.datetime.now()
        mpjpe = eval_result['mpjpe']
        pa_mpjpe = eval_result['pa_mpjpe']
        checkpoint_filename = os.path.abspath(
            os.path.join(self.save_dir, f'epoch:{epoch}_' + f'step:{total_step_count}_' + timestamp.strftime('%Y_%m_%d-%H_%M_%S')
                         + f'_{mpjpe}_{pa_mpjpe}' + '.pkl'))
        checkpoint = {}
        for model in models:
            checkpoint[model] = models[model].state_dict()
        for optimizer in optimizers:
            checkpoint[optimizer] = optimizers[optimizer].state_dict()
        checkpoint['epoch'] = epoch
        checkpoint['batch_idx'] = batch_idx
        checkpoint['batch_size'] = batch_size
        # checkpoint['dataset_perm'] = dataset_perm
        checkpoint['total_step_count'] = total_step_count
        print(timestamp, 'Epoch:', epoch, 'Iteration:', batch_idx)
        print('Saving checkpoint file [' + checkpoint_filename + ']')
        jittor.save(checkpoint, checkpoint_filename)
        return

    def load_checkpoint(self, models, optimizers, checkpoint_file=None):
        if checkpoint_file is None:
            print('Loading latest checkpoint [' + self.latest_checkpoint + ']')
            checkpoint_file = self.latest_checkpoint
        checkpoint = jittor.load(checkpoint_file)
        for model in models:
            if model in checkpoint:
                models[model].load_state_dict(checkpoint[model])
        for optimizer in optimizers:
            if optimizer in checkpoint:
                optimizers[optimizer].load_state_dict(checkpoint[optimizer])
        return {'epoch': checkpoint['epoch'],
                'batch_idx': checkpoint['batch_idx'],
                'batch_size': checkpoint['batch_size'],
                # 'dataset_perm': checkpoint['dataset_perm'],
                'total_step_count': checkpoint['total_step_count']}

    def _get_latest_checkpoint(self):
        """
        get latest checkpoint
        """
        checkpoint_list = []
        for dirpath, dirnames, filenames in os.walk(self.save_dir):
            for filename in filenames:
                if filename.endswith('.pkl'):
                    checkpoint_list.append(os.path.abspath(os.path.join(dirpath, filename)))
        checkpoint_list = sorted(checkpoint_list)
        self.latest_checkpoint = None if (len(checkpoint_list) is 0) else checkpoint_list[-1]
        return


