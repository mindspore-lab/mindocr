import os
import time

from packaging import version

import mindspore as ms
from mindspore import save_checkpoint
from mindspore.train.callback._callback import Callback, _handle_loss

from .checkpoint import CheckpointManager
from .evaluator import Evaluator
from .logger import Logger
from .misc import AverageMeter
from .recorder import PerfRecorder

__all__ = ["EvalSaveCallback"]

# WARNING: `mindspore.ms_function` will be deprecated and removed in a future version.
if version.parse(ms.__version__) >= version.parse("2.0.0rc"):
    from mindspore import jit
else:
    from mindspore import ms_function

    jit = ms_function


class EvalSaveCallback(Callback):
    """
    Callbacks for evaluation while training

    Args:
        network (nn.Cell): network (without loss)
        loader (Dataset): dataloader
        ema: if not None, the ema params will be loaded to the network for evaluation.
    """

    def __init__(
        self,
        network,
        loader=None,
        loss_fn=None,
        postprocessor=None,
        metrics=None,
        pred_cast_fp32=False,
        rank_id=0,
        device_num=None,
        logger=None,
        batch_size=20,
        ckpt_save_dir="./",
        main_indicator="hmean",
        ema=None,
        loader_output_columns=[],
        input_indices=None,
        label_indices=None,
        meta_data_indices=None,
        val_interval=1,
        val_start_epoch=1,
        log_interval=1,
        ckpt_save_policy="top_k",
        ckpt_max_keep=10,
        start_epoch=0,
    ):
        self.rank_id = rank_id
        self.is_main_device = rank_id in [0, None]
        self.loader_eval = loader
        self.network = network
        self.ema = ema
        self.logger = Logger("mindocr")
        self.val_interval = val_interval
        self.val_start_epoch = val_start_epoch
        self.log_interval = log_interval
        self.batch_size = batch_size
        if self.loader_eval is not None:
            self.net_evaluator = Evaluator(
                network,
                loader,
                loss_fn,
                postprocessor,
                metrics,
                pred_cast_fp32=pred_cast_fp32,
                loader_output_columns=loader_output_columns,
                input_indices=input_indices,
                label_indices=label_indices,
                meta_data_indices=meta_data_indices,
            )
            self.main_indicator = main_indicator
            self.best_perf = -1e8
        else:
            self.main_indicator = "train_loss"
            self.best_perf = 1e8

        self.ckpt_save_dir = ckpt_save_dir
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)

        self.last_epoch_end_time = time.time()
        self.epoch_start_time = time.time()
        self.step_start_time = time.time()

        self._loss_avg_meter = AverageMeter()

        self._reduce_sum = ms.ops.AllReduce()
        self._device_num = device_num
        # lamda expression is not supported in jit
        self._loss_reduce = self._reduce if device_num is not None else lambda x: x

        if self.is_main_device:
            self.ckpt_save_policy = ckpt_save_policy
            self.ckpt_manager = CheckpointManager(
                ckpt_save_dir,
                ckpt_save_policy,
                k=ckpt_max_keep,
                prefer_low_perf=(self.main_indicator == "train_loss"),
            )
        self.start_epoch = start_epoch

    @jit
    def _reduce(self, x):
        return self._reduce_sum(x) / self._device_num  # average value across all devices

    def on_train_step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        loss = _handle_loss(cb_params.net_outputs)
        cur_epoch = cb_params.cur_epoch_num
        data_sink_mode = cb_params.dataset_sink_mode
        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        self._loss_avg_meter.update(self._loss_reduce(loss))

        if not data_sink_mode and cur_step_in_epoch % self.log_interval == 0:
            opt = cb_params.train_network.optimizer
            learning_rate = opt.learning_rate
            cur_lr = learning_rate(opt.global_step - 1).asnumpy().squeeze()
            per_step_time = (time.time() - self.step_start_time) * 1000 / self.log_interval
            fps = self.batch_size * 1000 / per_step_time
            loss = self._loss_avg_meter.val.asnumpy()
            msg = (
                f"epoch: [{cur_epoch}/{cb_params.epoch_num}] step: [{cur_step_in_epoch}/{cb_params.batch_num}], "
                f"loss: {loss:.6f}, lr: {cur_lr:.6f}, per step time: {per_step_time:.3f} ms, fps: {fps:.2f} img/s"
            )
            self.logger.info(msg)
            self.step_start_time = time.time()

    def on_train_epoch_begin(self, run_context):
        """
        Called before each epoch beginning.
        Args:
            run_context (RunContext): Include some information of the model.
        """
        self._loss_avg_meter.reset()
        self.epoch_start_time = time.time()
        self.step_start_time = time.time()

    def on_train_epoch_end(self, run_context):
        """
        Called after each training epoch end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        cur_epoch = cb_params.cur_epoch_num
        train_time = time.time() - self.epoch_start_time
        train_loss = self._loss_avg_meter.avg.asnumpy()

        data_sink_mode = cb_params.dataset_sink_mode
        if data_sink_mode:
            loss_scale_manager = cb_params.train_network.network.loss_scaling_manager
        else:
            loss_scale_manager = cb_params.train_network.loss_scaling_manager

        epoch_time = time.time() - self.epoch_start_time
        per_step_time = epoch_time * 1000 / cb_params.batch_num
        fps = 1000 * self.batch_size / per_step_time
        msg = (
            f"epoch: [{cur_epoch}/{cb_params.epoch_num}], loss: {train_loss:.6f}, "
            f"epoch time: {epoch_time:.3f} s, per step time: {per_step_time:.3f} ms, fps: {fps:.2f} img/s"
        )
        self.logger.info(msg)

        eval_done = False
        if self.loader_eval is not None:
            if cur_epoch >= self.val_start_epoch and (cur_epoch - self.val_start_epoch) % self.val_interval == 0:
                eval_start = time.time()
                if self.ema is not None:
                    # swap ema weight and network weight
                    self.ema.swap_before_eval()
                measures = self.net_evaluator.eval()

                eval_done = True
                if self.is_main_device:
                    perf = measures[self.main_indicator]
                    eval_time = time.time() - eval_start
                    self.logger.info(f"Performance: {measures}, eval time: {eval_time}")
            else:
                measures = {m_name: None for m_name in self.net_evaluator.metric_names}
                eval_time = 0
                perf = 1e-8
        else:
            perf = train_loss

        # save best models and results using card 0
        if self.is_main_device:
            # save best models
            if (self.main_indicator == "train_loss" and perf < self.best_perf) or (
                self.main_indicator != "train_loss" and eval_done and perf > self.best_perf
            ):  # when val_while_train enabled, only find best checkpoint after eval done.
                self.best_perf = perf
                # ema weight will be saved if enabled.
                save_checkpoint(self.network, os.path.join(self.ckpt_save_dir, "best.ckpt"))

                self.logger.info(f"=> Best {self.main_indicator}: {self.best_perf}, checkpoint saved.")

            # save history checkpoints
            self.ckpt_manager.save(self.network, perf, ckpt_name=f"e{cur_epoch}.ckpt")
            ms.save_checkpoint(
                cb_params.train_network,
                os.path.join(self.ckpt_save_dir, "train_resume.ckpt"),
                append_dict={"epoch_num": cur_epoch, "loss_scale": loss_scale_manager.get_loss_scale()},
            )
            # record results
            if cur_epoch == 1:
                if self.loader_eval is not None:
                    perf_columns = ["loss"] + list(measures.keys()) + ["train_time", "eval_time"]
                else:
                    perf_columns = ["loss", "train_time"]
                self.rec = PerfRecorder(self.ckpt_save_dir, metric_names=perf_columns)  # record column names
            elif cur_epoch == self.start_epoch + 1:
                self.rec = PerfRecorder(self.ckpt_save_dir, resume=True)

            if self.loader_eval is not None:
                epoch_perf_values = [cur_epoch, train_loss] + list(measures.values()) + [train_time, eval_time]
            else:
                epoch_perf_values = [cur_epoch, train_loss, train_time]
            self.rec.add(*epoch_perf_values)  # record column values

        # swap back network weight and ema weight. MUST execute after model saving and before next-step training
        if (self.ema is not None) and eval_done:
            self.ema.swap_after_eval()

        # tot_time = time.time() - self.last_epoch_end_time
        self.last_epoch_end_time = time.time()

    def on_train_end(self, run_context):
        if self.is_main_device:
            self.rec.save_curves()  # save performance curve figure
            self.logger.info(f"=> Best {self.main_indicator}: {self.best_perf} \nTraining completed!")

            if self.ckpt_save_policy == "top_k":
                log_str = f"Top K checkpoints:\n{self.main_indicator}\tcheckpoint\n"
                for p, ckpt_name in self.ckpt_manager.get_ckpt_queue():
                    log_str += f"{p:.4f}\t{os.path.join(self.ckpt_save_dir, ckpt_name)}\n"
                self.logger.info(log_str)
