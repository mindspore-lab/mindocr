import os
import time
from tqdm import tqdm
from typing import List
from packaging import version

import mindspore as ms
from mindspore.ops import functional as F
from mindspore.common import dtype as mstype
from mindspore import save_checkpoint
from mindspore.train.callback._callback import Callback, _handle_loss
from .visualize import draw_bboxes, show_imgs, recover_image
from .recorder import PerfRecorder

__all__ = ['Evaluator', 'EvalSaveCallback']

# WARNING: `mindspore.ms_function` will be deprecated and removed in a future version.
if version.parse(ms.__version__) >= version.parse('2.0.0rc'):
    from mindspore import jit
else:
    from mindspore import ms_function
    jit = ms_function


class Evaluator:
    """
    Args:
        network: network
        dataloader : data loader to generate batch data, where the data columns in a batch are defined by the transform pipeline and `output_columns`.
        loss_fn: loss function
        postprocessor: post-processor
        metrics: metrics to evaluate network performance
        pred_cast_fp32: whehter to cast network prediction to float 32. Set True if AMP is used.
        input_indices: The indices of the data tuples which will be fed into the network. If it is None, then the first item will be fed only.
        label_indices: The indices of the data tuples which will be marked as label. If it is None, then the remaining items will be marked as label.
        meta_data_indices: The indices for the data tuples which will be marked as meta data. If it is None, then the item indices not in input or label indices are marked as meta data.
    """

    def __init__(self,
                 network,
                 dataloader,
                 loss_fn=None,
                 postprocessor=None,
                 metrics=None,
                 pred_cast_fp32=False,
                 input_indices=None,
                 label_indices=None,
                 meta_data_indices=None,
                 num_epochs=-1,
                 visualize=False,
                 verbose=False,
                 **kwargs):
        self.net = network
        self.postprocessor = postprocessor
        self.metrics = metrics if isinstance(metrics, List) else [metrics]
        self.metric_names = []
        for m in metrics:
            assert hasattr(m, 'metric_names') and isinstance(m.metric_names,
                                                             List), f'Metric object must contain `metric_names` attribute to indicate the metric names as a List type, but not found in {m.__class__.__name__}'
            self.metric_names += m.metric_names

        self.pred_cast_fp32 = pred_cast_fp32
        self.visualize = visualize
        self.verbose = verbose
        eval_loss = False
        if loss_fn is not None:
            eval_loss = True
            self.loss_fn = loss_fn
        assert eval_loss == False, 'not impl'

        # create iterator
        self.reload(dataloader, input_indices, label_indices, meta_data_indices, num_epochs)

    def reload(self, dataloader, input_indices=None, label_indices=None, meta_data_indices=None, num_epochs=-1):
        # create iterator
        self.iterator = dataloader.create_tuple_iterator(num_epochs=num_epochs, output_numpy=False, do_copy=False)
        self.num_batches_eval = dataloader.get_dataset_size()

        # dataset output columns
        self.input_indices = input_indices
        self.label_indices = label_indices
        self.meta_data_indices = meta_data_indices

    def eval(self):
        """
        Args:
        """
        eval_res = {}

        self.net.set_train(False)
        for m in self.metrics:
            m.clear()

        for i, data in tqdm(enumerate(self.iterator), total=self.num_batches_eval):
            if self.input_indices is not None:
                inputs = [data[x] for x in self.input_indices]
            else:
                inputs = [data[0]]

            if self.label_indices is not None:
                gt = [data[x] for x in self.label_indices]
            else:
                gt = data[1:]

            net_preds = self.net(*inputs)

            if self.pred_cast_fp32:
                if isinstance(net_preds, ms.Tensor):
                    net_preds = F.cast(net_preds, mstype.float32)
                else:
                    net_preds = [F.cast(p, mstype.float32) for p in net_preds]

            if self.postprocessor is not None:
                # additional info such as image path, original image size, pad shape, extracted in data processing
                if self.meta_data_indices is not None:
                    meta_info = [data[x] for x in self.meta_data_indices]
                else:
                    # assume the indices not in input_indices or label_indices are all meta_data_indices
                    input_indices = set(self.input_indices) if self.input_indices is not None else {0}
                    label_indices = set(self.label_indices) if self.label_indices is not None else set(range(1, len(data), 1))
                    meta_data_indices = sorted(set(range(len(data))) - input_indices - label_indices)
                    meta_info = [data[x] for x in meta_data_indices]

                data_info = {'labels': gt, 'img_shape': inputs[0].shape, 'meta_info': meta_info}
                preds = self.postprocessor(net_preds, **data_info)

            # metric internal update
            for m in self.metrics:
                m.update(preds, gt)

            # visualize
            if self.verbose:
                print('Data meta info: ', data_info)

            if self.visualize:
                img = img[0].asnumpy()
                assert ('polys' in preds) or ('polygons' in preds), 'Only support detection'
                gt_img_polys = draw_bboxes(recover_image(img), gt[0].asnumpy())
                pred_img_polys = draw_bboxes(recover_image(img), preds['polygons'].asnumpy())
                show_imgs([gt_img_polys, pred_img_polys], show=False, save_path=f'results/det_vis/gt_pred_{i}.png')

        for m in self.metrics:
            res_dict = m.eval()
            eval_res.update(res_dict)

        self.net.set_train(True)

        return eval_res


class EvalSaveCallback(Callback):
    """
    Callbacks for evaluation while training

    Args:
        network (nn.Cell): network (without loss)
        loader (Dataset): dataloader
        ema: if not None, the ema params will be loaded to the network for evaluation.
    """

    def __init__(self,
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
                 ckpt_save_dir='./',
                 main_indicator='hmean',
                 ema=None,
                 input_indices=None,
                 label_indices=None,
                 meta_data_indices=None,
                 val_interval=1,
                 val_start_epoch=1,
                 log_interval=1,
                 ):
        self.rank_id = rank_id
        self.is_main_device = rank_id in [0, None]
        self.loader_eval = loader
        self.network = network
        self.ema = ema
        self.logger = print if logger is None else logger.info
        self.val_interval = val_interval
        self.val_start_epoch = val_start_epoch
        self.log_interval = log_interval
        self.batch_size = batch_size
        if self.loader_eval is not None:
            self.net_evaluator = Evaluator(network, loader, loss_fn, postprocessor, metrics,
                                           pred_cast_fp32=pred_cast_fp32, input_indices=input_indices,
                                           label_indices=label_indices, meta_data_indices=meta_data_indices)
            self.main_indicator = main_indicator
            self.best_perf = -1e8
        else:
            self.main_indicator = 'train_loss'
            self.best_perf = 1e8

        self.ckpt_save_dir = ckpt_save_dir
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)

        self.last_epoch_end_time = time.time()
        self.epoch_start_time = time.time()
        self.step_start_time = time.time()

        self._losses = []

        self._reduce_sum = ms.ops.AllReduce()
        self._device_num = device_num
        # lamda expression is not supported in jit
        self._loss_reduce = self._reduce if device_num is not None else lambda x: x

    @jit
    def _reduce(self, x):
        return self._reduce_sum(x) / self._device_num   # average value across all devices

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

        self._losses.append(self._loss_reduce(loss))

        if not data_sink_mode and cur_step_in_epoch % self.log_interval == 0:
            opt = cb_params.train_network.optimizer
            learning_rate = opt.learning_rate
            cur_lr = learning_rate(opt.global_step - 1).asnumpy().squeeze()
            per_step_time = (time.time() - self.step_start_time) * 1000 / self.log_interval
            fps = self.batch_size * 1000 / per_step_time
            loss = self._losses[-1].asnumpy()
            msg = f"epoch: [{cur_epoch}/{cb_params.epoch_num}] step: [{cur_step_in_epoch}/{cb_params.batch_num}], " \
                  f"loss: {loss:.6f}, lr: {cur_lr:.6f}, per step time: {per_step_time:.3f} ms, fps: {fps:.2f} img/s"
            self.logger(msg)
            self.step_start_time = time.time()

    def on_train_epoch_begin(self, run_context):
        """
        Called before each epoch beginning.
        Args:
            run_context (RunContext): Include some information of the model.
        """
        self._losses.clear()
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
        train_time = (time.time() - self.epoch_start_time)
        train_loss = ms.ops.stack(self._losses).mean().asnumpy()

        epoch_time = (time.time() - self.epoch_start_time)
        per_step_time = epoch_time * 1000 / cb_params.batch_num
        fps = 1000 * self.batch_size / per_step_time
        msg = f"epoch: [{cur_epoch}/{cb_params.epoch_num}], loss: {train_loss:.6f}, " \
              f"epoch time: {epoch_time:.3f} s, per step time: {per_step_time:.3f} ms, fps: {fps:.2f} img/s"
        self.logger(msg)

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
                    self.logger(f'Performance: {measures}, eval time: {eval_time}')
            else:
                measures = {m_name: None for m_name in self.net_evaluator.metric_names}
                eval_time = 0
                perf = 1e-8
        else:
            perf = train_loss

        # save best models and results using card 0
        if self.is_main_device:
            # save best models
            if (self.main_indicator == 'train_loss' and perf < self.best_perf) \
                    or (
                    self.main_indicator != 'train_loss' and eval_done and perf > self.best_perf):  # when val_while_train enabled, only find best checkpoint after eval done.
                self.best_perf = perf
                # ema weight will be saved if enable.
                save_checkpoint(self.network, os.path.join(self.ckpt_save_dir, 'best.ckpt'))

                self.logger(f'=> Best {self.main_indicator}: {self.best_perf}, checkpoint saved.')

            # record results
            if cur_epoch == 1:
                if self.loader_eval is not None:
                    perf_columns = ['loss'] + list(measures.keys()) + ['train_time', 'eval_time']
                else:
                    perf_columns = ['loss', 'train_time']
                self.rec = PerfRecorder(self.ckpt_save_dir, metric_names=perf_columns)  # record column names

            if self.loader_eval is not None:
                epoch_perf_values = [cur_epoch, train_loss] + list(measures.values()) + [train_time, eval_time]
            else:
                epoch_perf_values = [cur_epoch, train_loss, train_time]
            self.rec.add(*epoch_perf_values)  # record column values

        # swap back network weight and ema weight. MUST execute after model saving and before next-step training
        if (self.ema is not None) and eval_done:
            self.ema.swap_after_eval()

        tot_time = time.time() - self.last_epoch_end_time
        self.last_epoch_end_time = time.time()

    def on_train_end(self, run_context):
        if self.is_main_device:
            self.rec.save_curves()  # save performance curve figure
            self.logger(f'=> Best {self.main_indicator}: {self.best_perf} \nTraining completed!')
