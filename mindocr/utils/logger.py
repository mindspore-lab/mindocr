"""Custom Logger."""
import logging
import os
import sys


class Logger(logging.Logger):
    """
    Logger.

    Args:
         logger_name: String. Logger name.
         rank: Integer. Rank id.
    """

    def __init__(self, logger_name, rank=0, log_fn=None):
        super(Logger, self).__init__(logger_name)
        self.rank = rank or 0
        self.log_fn = log_fn
        is_main_device = not rank

        if is_main_device:
            console = logging.StreamHandler(sys.stdout)
            console.setLevel(logging.INFO)
            formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
            console.setFormatter(formatter)
            self.addHandler(console)

    def setup_logging_file(self, log_dir):
        """Setup logging file."""
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, exist_ok=True)
        if self.log_fn is None:
            log_name = "log_%s.txt" % self.rank
            self.log_save_path = os.path.join(log_dir, log_name)
        else:
            self.log_save_path = os.path.join(log_dir, self.log_fn)
        fh = logging.FileHandler(self.log_save_path)
        fh.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s:%(levelname)s:%(message)s")
        fh.setFormatter(formatter)
        self.addHandler(fh)

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO):
            self._log(logging.INFO, msg, args, **kwargs)

    def save_args(self, args):
        self.info("Args:")
        args_dict = vars(args)
        for key in args_dict.keys():
            self.info("--> %s: %s", key, args_dict[key])
        self.info("")

    def important_info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO) and self.rank == 0:
            line_width = 2
            important_msg = "\n"
            important_msg += ("*" * 70 + "\n") * line_width
            important_msg += ("*" * line_width + "\n") * 2
            important_msg += "*" * line_width + " " * 8 + msg + "\n"
            important_msg += ("*" * line_width + "\n") * 2
            important_msg += ("*" * 70 + "\n") * line_width
            self.info(important_msg, *args, **kwargs)


def get_logger(log_dir, rank, log_fn=None):
    """Get Logger."""
    logger = Logger("mindocr", rank, log_fn=log_fn)
    logger.setup_logging_file(log_dir)

    return logger
