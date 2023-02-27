import time
import tqdm

import mindspore as ms
from mindspore.train.callback import Callback

__all__ = ['Evaluator', 'EvalCallback']

class Evaluator(object):
    def __init__(self, network, loss_fn=None, postprocessor=None, metrics=None, **kwargs):
        self.net = network
        self.loss_fn = loss_fn
        self.postprocessor = postprocessor
        self.metrics = metrics
    
    def eval(self, dataloader, num_keys_to_net=1, num_keys_of_labels=None, visualize=False):
        '''
        Args:
            dataloader (Dataset): data iterator which generates tuple of Tensor defined by the transform pipeline and 'output_keys'
        '''
        #for i, batch in tqdm(enumerate(dataloader.create_dict_iterator(num_epochs=1))):
        self.net.set_train(False)
        iterator = dataloader.create_tuple_iterator(num_epochs=1, output_numpy=False, do_copy=False)
        for i, data in tqdm(enumerate(iterator)):
            #start = time.time()
            # assume the first element is image
            img = data[0] #ms.Tensor(batch[0])
            #net_inputs = data[:num_keys_to_net]

            # TODO: if network input is not just an image.
            # network preds is a dict. for text det {'binary':, ...},  for text rec, {'head_out': }
            preds = self.net(img) 
            #preds = self.net(*net_inputs) # head output is dict. for text det {'binary', ...},  for text rec, {'head_out': }
            print('net predictions', preds)

            if self.postprocessor is not None:
                preds = self.postprocessor(preds) # for text det, res = {'polygons':, 'scores':}

            print('postproc output:', preds)
            print('labels: ', data[1:])

            #cur_time = time.time() - start
            #raw_metric = self.metric.validate_measure(batch, (boxes, scores))

            #cur_frame = img.shape[0]

            #raw_metric, (cur_frame, cur_time) = self.once_eval(batch)
            #raw_metrics.append(raw_metric)
            #if count:
            #    total_frame += cur_frame
            #    total_time += cur_time

            #count += 1
            '''
            if show_imgs:
                img = batch['original'].squeeze().astype('uint8')
                # gt
                for idx, poly in enumerate(raw_metric['gt_polys']):
                    poly = np.expand_dims(poly, -2).astype(np.int32)
                    if idx in raw_metric['gt_dont_care']:
                        cv2.polylines(img, [poly], True, (255, 160, 160), 4)
                    else:
                        cv2.polylines(img, [poly], True, (255, 0, 0), 4)
                # pred
                for idx, poly in enumerate(raw_metric['det_polys']):
                    poly = np.expand_dims(poly, -2).astype(np.int32)
                    if idx in raw_metric['det_dont_care']:
                        cv2.polylines(img, [poly], True, (200, 255, 200), 4)
                    else:
                        cv2.polylines(img, [poly], True, (0, 255, 0), 4)
                if not os.path.exists(self.config.eval.image_dir):
                    os.makedirs(self.config.eval.image_dir)
                cv2.imwrite(self.config.eval.image_dir + f'eval_{count}.jpg', img)
            '''

        #metrics = self.metric.gather_measure(raw_metrics)
        #fps = total_frame / total_time

        # TODO: do it outside
        self.net.set_train(True)
        return None #metrics, fps
        


class EvalCallback(Callback):
    '''
    Callbacks for evaluation while training

    Args:
        network (nn.Cell): network (without loss)
        loader (Dataset): dataloader
    '''
    def __init__(self, 
                network, 
                loader, 
                loss_fn=None, 
                postprocessor=None, 
                metrics=None, 
                rank_id=None):
        self.rank_id = rank_id
        if rank_id in [None, 0]:
            self.net_eval = Evaluator(network, loss_fn, postprocessor, metrics)
            self.loader_eval = loader

    def __enter__(self):
        pass

    def __exit__(self, *exc_args):
        pass

    def on_train_step_begin(self, run_context):
        self.step_start_time = time.time()
        print('Train step begin -------------- ')

    def on_train_step_end(self, run_context):
        """
        Print training loss at the end of step.

        Args:
            run_context (RunContext): Context of the train running.
        """
        cb_params = run_context.original_args()
        loss = cb_params.net_outputs
        cur_epoch = cb_params.cur_epoch_num
        data_sink_mode = cb_params.dataset_sink_mode
        '''
        if cb_params.net_outputs is not None:
            if isinstance(loss, tuple):
                if loss[1]:
                    self.config.logger.info("==========overflow!==========")
                loss = loss[0]
            loss = loss.asnumpy()
        else:
            self.config.logger.info("custom loss callback class loss is None.")
            return

        cur_step_in_epoch = (cb_params.cur_step_num - 1) % cb_params.batch_num + 1

        if cur_step_in_epoch == 1:
            self.loss_avg = AverageMeter()
        self.loss_avg.update(loss)

        if isinstance(loss, float) and (np.isnan(loss) or np.isinf(loss)):
            raise ValueError(
                "epoch: {} step: {}. Invalid loss, terminating training.".format(cur_epoch, cur_step_in_epoch))
        if self._per_print_times != 0 and (
                cb_params.cur_step_num - self._last_print_time) >= self._per_print_times and not data_sink_mode:
            self._last_print_time = cb_params.cur_step_num
            loss_log = "epoch: [%s/%s] step: [%s/%s], loss: %.6f, lr: %.6f, per step time: %.3f ms" % (
                cur_epoch, self.config.train.total_epochs, cur_step_in_epoch, self.config.steps_per_epoch,
                np.mean(self.loss_avg.avg), self.lr[self.cur_steps], (time.time() - self.step_start_time) * 1000)
            self.config.logger.info(loss_log)
        self.cur_steps += 1
        '''

    def on_train_epoch_begin(self, run_context):
        """
        Called before each epoch beginning.
        Args:
            run_context (RunContext): Include some information of the model.
        """
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
        epoch_time = (time.time() - self.epoch_start_time)
        #loss_log = "epoch: [%s/%s], loss: %.6f, epoch time: %.3f s" % (
        #    cur_epoch, self.config.train.total_epochs, loss[0].asnumpy(), epoch_time,
        #    epoch_time * 1000 / self.config.steps_per_epoch)
        loss_log = "epoch: [%s], loss: %.6f, epoch time: %.3f s" % (
            cur_epoch, loss.asnumpy(), epoch_time)
        print('INFO: ', loss_log)

        if self.rank_id in [0, None]:
            metrics = self.net_eval(self.loader_eval) 

    def on_train_end(self, run_context):
        if self.rank_id == 0:
            print('INFO: best fmeasure is: %s' % self.max_f)

