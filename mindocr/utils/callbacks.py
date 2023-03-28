import os
import time
from tqdm import tqdm
from typing import List
import shutil

import numpy as np
import mindspore as ms
from mindspore import save_checkpoint
from mindspore.train.callback._callback import Callback, _handle_loss
from mindocr.utils.visualize import draw_bboxes, show_imgs, recover_image
from mindocr.utils.recorder import PerfRecorder

__all__ = ['Evaluator', 'EvalSaveCallback']


class Evaluator:
    """
    Args:
        metric:
    """
    def __init__(self, network, loss_fn=None, postprocessor=None, metrics=None, visualize=False, verbose=False,
                 **kwargs):
        self.net = network
        self.postprocessor = postprocessor
        self.metrics = metrics if isinstance(metrics, List) else [metrics]
        self.visualize = visualize
        self.verbose = verbose
        eval_loss = False
        if loss_fn is not None:
            eval_loss = True
            self.loss_fn = loss_fn
        # TODO: add support for computing evaluation loss
        assert eval_loss == False, 'not impl'

    def eval(self, dataloader, num_columns_to_net=1, num_keys_of_labels=None):
        """
        Args:
            dataloader (Dataset): data iterator which generates tuple of Tensor defined by the transform pipeline and 'output_columns'
        """
        eval_res = {}

        self.net.set_train(False)
        iterator = dataloader.create_tuple_iterator(num_epochs=1, output_numpy=False, do_copy=False)
        for m in self.metrics:
            m.clear()

        for i, data in tqdm(enumerate(iterator), total=dataloader.get_dataset_size()):
            # start = time.time()
            # TODO: if network input is not just an image.
            # assume the first element is image
            img = data[0]  # ms.Tensor(batch[0])
            gt = data[1:]  # ground truth,  (polys, ignore_tags) for det,

            # TODO: in graph mode, the output dict is somehow converted to tuple. Is the order of in tuple the same as dict binary, thresh, thres_binary? to check
            # {'binary':, 'thresh: ','thresh_binary': } for text detect; {'head_out': } for text rec
            net_preds = self.net(img)
            # net_inputs = data[:num_columns_to_net]
            # gt = data[num_columns_to_net:] # ground truth
            # preds = self.net(*net_inputs) # head output is dict. for text det {'binary', ...},  for text rec, {'head_out': }
            # print('net predictions', preds)

            if self.postprocessor is not None:
                preds = self.postprocessor(net_preds)  # {'polygons':, 'scores':} for text det

            # metric internal update
            for m in self.metrics:
                m.update(preds, gt)

            if self.verbose:
                if isinstance(net_preds, tuple):
                    print('pred binary map:', net_preds[0].shape, net_preds[0].max(), net_preds[0].min())
                    print('thresh binary map:', net_preds[2].shape, net_preds[2].max(), net_preds[2].min())
                else:
                    print('pred binary map:', net_preds['binary'].shape, net_preds['binary'].max(),
                          net_preds['binary'].min())
                    print('thresh binary map:', net_preds['thresh_binary'].shape, net_preds['thresh_binary'].max(),
                          net_preds['binary'].min())
                print('pred polys:', preds['polygons'])

            if self.visualize:
                img = img[0].asnumpy()
                assert ('polys' in preds) or ('polygons' in preds), 'Only support detection'
                gt_img_polys = draw_bboxes(recover_image(img), gt[0].asnumpy())
                pred_img_polys = draw_bboxes(recover_image(img), preds['polygons'].asnumpy())
                show_imgs([gt_img_polys, pred_img_polys], show=False, save_path=f'results/det_vis/gt_pred_{i}.png')

        for m in self.metrics:
            res_dict = m.eval()
            eval_res.update(res_dict)
        # fps = total_frame / total_time

        self.net.set_train(True)

        return eval_res


class EvalSaveCallback(Callback):
    """
    Callbacks for evaluation while training

    Args:
        network (nn.Cell): network (without loss)
        loader (Dataset): dataloader
        saving_config (dict):
    """
    def __init__(self,
                 network,
                 loader=None,
                 loss_fn=None,
                 postprocessor=None,
                 metrics=None,
                 rank_id=0,
                 ckpt_save_dir='./',
                 main_indicator='hmean',
                 ):
        self.rank_id = rank_id
        self.is_main_device = rank_id in [0, None]
        self.loader_eval = loader
        self.network = network
        if self.loader_eval is not None:
            self.net_evaluator = Evaluator(network, loss_fn, postprocessor, metrics)
            self.main_indicator = main_indicator
            self.best_perf = -1e8
        else:
            self.main_indicator = 'train_loss'
            self.best_perf = 1e8

        self.ckpt_save_dir = ckpt_save_dir
        if not os.path.exists(ckpt_save_dir):
            os.makedirs(ckpt_save_dir)

        self.last_epoch_end_time = time.time()

    def on_train_step_begin(self, run_context):
        self.step_start_time = time.time()

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

        # TODO: need to stop gradient here ?
        self._losses.append(loss.asnumpy())

    def on_train_epoch_begin(self, run_context):
        """
        Called before each epoch beginning.
        Args:
            run_context (RunContext): Include some information of the model.
        """
        self._losses = []
        self.epoch_start_time = time.time()

    def on_train_epoch_end(self, run_context):
        """
        Called after each training epoch end.

        Args:
            run_context (RunContext): Include some information of the model.
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        cur_epoch = cb_params.cur_epoch_num
        train_time = (time.time() - self.epoch_start_time)
        train_loss = np.average(self._losses) # TODO: aggregate training loss for multiple cards

        print(
            f"Epoch: {cur_epoch}, "
            f"loss:{train_loss:.5f}, training time:{train_time:.3f}s"
        )

        if self.loader_eval is not None:
                eval_start = time.time()
                measures = self.net_evaluator.eval(self.loader_eval)
                if self.is_main_device:
                    perf = measures[self.main_indicator]
                    eval_time = time.time() - eval_start
                    print(f'Performance: {measures}, eval time: {eval_time}')
        else:
            perf = train_loss

        # save models and results using card 0
        if self.is_main_device:
            if (perf > self.best_perf) ^ (self.main_indicator=='train_loss'):
                self.best_perf = perf 
                save_checkpoint(self.network, os.path.join(self.ckpt_save_dir, 'best.ckpt'))
                print(f'=> Best {self.main_indicator}: {self.best_perf}, checkpoint saved.')

            # record results
            if cur_epoch == 1:
                metric_names = ['loss']
                if self.loader_eval is not None:
                    metric_names +=  list(measures.keys()) + ['train_time', 'eval_time']
                else:
                    metric_names +=  ['train_time']

                self.rec = PerfRecorder(self.ckpt_save_dir, metric_names=metric_names)
            epoch_metric_values = [cur_epoch, train_loss]
            if self.loader_eval is not None:
                epoch_metric_values += list(measures.values()) + [train_time, eval_time]
            else:
                epoch_metric_values += [train_time]
            self.rec.add(*epoch_metric_values)

        tot_time = time.time()-self.last_epoch_end_time 
        self.last_epoch_end_time = time.time()
        print(f'Total time cost for epoch {cur_epoch}: {tot_time}')

    def on_train_end(self, run_context):
        if self.is_main_device:
            self.rec.save_curves()  # save performance curve figure
            print(f'=> Best {self.main_indicator}: {self.best_perf} \nTraining completed!')


class LossCallBack(Callback):
    """
    Monitor the loss in training.
    If the loss in NAN or INF terminating training.
    """

    def __init__(self, epoch_size, batch_size, logger, per_print_times=1, global_steps=0):
        super(LossCallBack, self).__init__()
        self.epoch_size = epoch_size
        self.batch_size = batch_size
        self.logger = logger
        self.global_steps = global_steps
        self._per_print_times = per_print_times
        self.step_start_time = time.time()
        self.epoch_start_time = time.time()

    def _handle_loss(self, net_outputs):
        """Handle loss"""
        if isinstance(net_outputs, (tuple, list)):
            if isinstance(net_outputs[0], ms.Tensor) and isinstance(net_outputs[0].asnumpy(), np.ndarray):
                loss = net_outputs[0].asnumpy()
        elif isinstance(net_outputs, ms.Tensor) and isinstance(net_outputs.asnumpy(), np.ndarray):
            loss = float(np.mean(net_outputs.asumpy()))
        return loss

    def on_train_epoch_begin(self, run_context):
        self.epoch_start_time = time.time()
        self.step_start_time = time.time()

    def on_train_step_end(self, run_context):
        cb_params = run_context.original_args()
        loss = self._handle_loss(cb_params.net_outputs)
        overflow = str(cb_params.net_outputs[1].asnumpy())
        cur_epoch = cb_params.cur_epoch_num
        data_sink_mode = cb_params.get('dataset_sink_mode', False)
        if not data_sink_mode:
            cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1
            if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
                raise ValueError(
                    "epoch: {} step: {}. Invalid loss, terminating training.".format(cur_epoch, cur_step_in_epoch))

            if cur_step_in_epoch % self._per_print_times == 0:
                per_step_time = (time.time() - self.step_start_time) * 1000 / self._per_print_times
                fps = self.batch_size * 1000 / per_step_time
                lr = cb_params.train_network.optimizer.learning_rate
                cur_lr = lr(ms.Tensor(self.global_steps)).asnumpy()
                msg = "epoch: [%s/%s] step: [%s/%s], loss: %.6f, overflow: %s, lr: %.6f, per step time: %.3f ms, fps: %.2f img/s" % (
                    cur_epoch, self.epoch_size, cur_step_in_epoch, cb_params.batch_num,
                    loss, overflow, cur_lr, per_step_time, fps)
                self.logger.info(msg)
                self.step_start_time = time.time()
        self.global_steps += 1

    def on_train_epoch_end(self, run_context):
        cb_params = run_context.original_args()
        loss = self._handle_loss(cb_params.net_outputs)
        cur_epoch_num = cb_params.cur_epoch_num
        epoch_time = time.time() - self.epoch_start_time
        per_step_time = epoch_time * 1000 / cb_params.batch_num
        fps = 1000 * self.batch_size / per_step_time
        msg = 'epoch: [%s/%s] loss: %.6f, epoch time: %.3f s, per step time: %.3f ms, fps: %.2f' % (
            cur_epoch_num, self.epoch_size, loss, epoch_time, per_step_time, fps)
        self.logger.info(msg)


class ResumeCallback(Callback):
    def __init__(self, start_epoch_num=0):
        super(ResumeCallback, self).__init__()
        self.start_epoch_num = start_epoch_num

    def on_train_epoch_begin(self, run_context):
        run_context.original_args().cur_epoch_num += self.start_epoch_num
