import logging
import os
from typing import List

import matplotlib.pyplot as plt
import numpy as np

import mindspore as ms

_logger = logging.getLogger(__name__)


def plot_result(result_path, save_fig=False, sep="\t", na_padding="last", replace_val=-1):
    metrics = {}
    with open(result_path) as fp:
        for i, line in enumerate(fp):
            if i == 0:
                attrs = line.strip().split(sep)
                for attr in attrs:
                    metrics[attr] = []
            else:
                vals = line.strip().split(sep)
                # epochs.append(vals[0])
                for j, val in enumerate(vals):
                    if not (val in ["NA", "None", "N/A", "null"]):
                        metrics[attrs[j]].append(float(val))
                    else:
                        if na_padding == "replace":
                            metrics[attrs[j]].append(replace_val)
                        elif na_padding == "last":
                            if len(metrics[attrs[j]]) == 0:
                                last_val = 0
                            else:
                                last_val = metrics[attrs[j]][-1]
                            # TODO: skip plotting the points with NA value
                            metrics[attrs[j]].append(last_val)
                        else:
                            raise ValueError

    epochs = metrics[attrs[0]]
    fig, axs = plt.subplots(len(attrs) - 1)
    for i, attr in enumerate(attrs[1:]):
        axs[i].plot(epochs, metrics[attr])
        axs[i].set_title(attr)
        axs[i].grid()

    if save_fig:
        save_path = result_path.replace(".log", ".pdf")
        plt.savefig(save_path)

    return epochs, metrics


class PerfRecorder(object):
    def __init__(
        self,
        save_dir,
        metric_names: List = ["loss", "precision", "recall", "hmean", "s/epoch"],
        file_name="result.log",
        separator="\t",
        resume=False,
    ):
        self.save_dir = save_dir
        self.sep = separator
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            _logger.info(f"{save_dir} not exist. Created.")

        self.log_txt_fp = os.path.join(save_dir, file_name)
        if not resume:
            result_log = separator.join(["Epoch"] + metric_names)
            with open(self.log_txt_fp, "w", encoding="utf-8") as fp:
                fp.write(result_log + "\n")

    def add(self, epoch, *measures):
        """
        measures (Tuple): measurement values corresponding to the metric names
        """
        sep = self.sep
        line = f"{epoch}{sep}"
        for i, m in enumerate(measures):
            if isinstance(m, ms.Tensor):
                m = m.asnumpy()

            if isinstance(m, float) or isinstance(m, np.float32):
                line += f"{m:.4f}"
            elif m is None:
                line += "NA"
            else:
                line += f"{m}"

            if i < len(measures) - 1:
                line += f"{sep}"
        # line += f"{epoch_time:.2f}\n"

        with open(self.log_txt_fp, "a", encoding="utf-8") as fp:
            fp.write(line + "\n")

    def save_curves(self):
        plot_result(self.log_txt_fp, save_fig=True, sep=self.sep)


if __name__ == "__main__":
    r = PerfRecorder("./")
    r.add(1, 0.2, 0.4, 0.5, 199)
