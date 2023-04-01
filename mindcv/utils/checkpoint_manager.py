"""checkpoint manager """
import os
import stat

import numpy as np

import mindspore as ms
from mindspore import log as logger


class CheckpointManager:
    """
    Manage checkpoint files according to ckpt_save_policy of checkpoint.
    Args:
        ckpt_save_policy (str): Checkpoint saving strategy. The optional values is None, "top_k" or "latest_k".
        None means to save each checkpoint, top_k means to save K checkpoints with the highest accuracy,
        and latest_k means saving the latest K checkpoint. Default: None.
    """

    def __init__(self, ckpt_save_policy=None):
        self._ckpoint_filelist = []
        self.ckpt_save_policy = ckpt_save_policy

    @property
    def ckpoint_filelist(self):
        """Get all the related checkpoint files managed here."""
        return self._ckpoint_filelist

    @property
    def ckpoint_num(self):
        """Get the number of the related checkpoint files managed here."""
        return len(self._ckpoint_filelist)

    def update_ckpoint_filelist(self, directory, prefix):
        """Update the checkpoint file list."""
        self._ckpoint_filelist = []
        files = os.listdir(directory)
        for filename in files:
            if os.path.splitext(filename)[-1] == ".ckpt" and filename.startswith(prefix + "-"):
                mid_name = filename[len(prefix) : -5]
                flag = not (True in [char.isalpha() for char in mid_name])
                if flag:
                    self._ckpoint_filelist.append(os.path.join(directory, filename))

    def remove_ckpoint_file(self, file_name):
        """Remove the specified checkpoint file from this checkpoint manager and also from the directory."""
        try:
            os.chmod(file_name, stat.S_IWRITE)
            os.remove(file_name)
        except OSError:
            logger.warning("OSError, failed to remove the older ckpt file %s.", file_name)
        except ValueError:
            logger.warning("ValueError, failed to remove the older ckpt file %s.", file_name)

    def remove_oldest_ckpoint_file(self):
        """Remove the oldest checkpoint file from this checkpoint manager and also from the directory."""
        ckpoint_files = sorted(self._ckpoint_filelist, key=os.path.getmtime)
        self.remove_ckpoint_file(ckpoint_files[0])
        self._ckpoint_filelist.remove(ckpoint_files[0])

    def keep_one_ckpoint_per_minutes(self, minutes, cur_time):
        """Only keep the latest one ckpt file per minutes, remove other files generated in [last_time, cur_time]."""
        del_list = []
        oldest_file = ""
        oldest_time = cur_time
        for ck_file in self._ckpoint_filelist:
            modify_time = os.path.getmtime(ck_file)
            if cur_time - modify_time < 60 * minutes:
                del_list.append(ck_file)

                if modify_time < oldest_time:
                    oldest_time = modify_time
                    oldest_file = ck_file

        for mv_file in del_list:
            if mv_file == oldest_file:
                continue
            self.remove_ckpoint_file(mv_file)

    def top_K_checkpoint(self, network, K=10, metric=None, save_path=""):
        """Save and return Top K checkpoint address and accuracy."""
        last_file = self._ckpoint_filelist[-1] if self._ckpoint_filelist else None
        if type(metric) is not np.ndarray:
            metric = metric.asnumpy()
        if self.ckpoint_num < K or np.greater(metric, last_file[1]):
            if self.ckpoint_num >= K:
                delete = K - 1
                if delete < 0 or self.ckpoint_num <= delete:
                    return
                to_delete = self._ckpoint_filelist[delete:]
                for d in to_delete:
                    self.remove_ckpoint_file(d[0])
                self._ckpoint_filelist = self._ckpoint_filelist[:delete]
            ms.save_checkpoint(network, save_path, async_save=True)
            self._ckpoint_filelist.append((save_path, float(metric)))
            self._ckpoint_filelist = sorted(self._ckpoint_filelist, key=lambda x: x[1], reverse=True)

    def latest_K_checkpoint(self, network, K=10, save_path=""):
        """Save latest K checkpoint."""
        if K and 0 < K <= self.ckpoint_num:
            self.remove_oldest_ckpoint_file()
        ms.save_checkpoint(network, save_path, async_save=True)
        self._ckpoint_filelist.append(save_path)

    def save_ckpoint(self, network, num_ckpt=10, metric=None, save_path=""):
        """Save checkpoint according to different save strategy."""
        if self.ckpt_save_policy is None:
            ms.save_checkpoint(network, save_path, async_save=True)
        elif self.ckpt_save_policy == "top_k":
            if metric is None:
                raise ValueError(f"The expected 'metric' is not None, but got: {metric}.")
            self.top_K_checkpoint(network, K=num_ckpt, metric=metric, save_path=save_path)
            return self._ckpoint_filelist
        elif self.ckpt_save_policy == "latest_k":
            self.latest_K_checkpoint(network, K=num_ckpt, save_path=save_path)
            return self._ckpoint_filelist
        else:
            raise ValueError(
                f"The expected 'ckpt_save_policy' is None, top_k or latest_k, but got: {self.ckpt_save_policy}."
            )
