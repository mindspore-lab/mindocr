import argparse
import logging
import os
import sys
import threading
import time
from logging.handlers import RotatingFileHandler

# Log level name and number mapping
_name_to_log_level = {
    'ERROR': 40,
    'WARNING': 30,
    'INFO': 20,
    'DEBUG': 10,
}

# mindspore level and level name
_ms_level_to_name = {
    '3': 'ERROR',
    '2': 'WARNING',
    '1': 'INFO',
    '0': 'DEBUG',
}

MAX_BYTES = 100 * 1024 * 1024
BACKUP_COUNT = 10
LOG_TYPE = "mindocr"
LOG_ENV = "MINDOCR_LOG_LEVEL"
INFER_INSTALL_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), '../../')) + "/"


class DataFormatter(logging.Formatter):
    """Log formatter"""

    def __init__(self, sub_module, fmt=None, **kwargs):
        """
        Initialization of logFormatter.
        :param sub_module: The submodule name.
        :param fmt: Specified format pattern. Default: None.
        :param kwargs: None
        """
        super(DataFormatter, self).__init__(fmt=fmt, **kwargs)
        self.sub_module = sub_module.upper()

    def formatTime(self, record, datefmt=None):
        """
        Override formatTime for uniform format %Y-%m-%d-%H:%M:%S.SSS.SSS
        :param record: Log record
        :param datefmt: Date format
        :return: formatted timestamp
        """
        create_time = self.converter(record.created)
        if datefmt:
            return time.strftime(datefmt, create_time)

        timestamp = time.strftime('%Y-%m-%d-%H:%M:%S', create_time)
        record_msecs = str(round(record.msecs * 1000))
        # Format the time stamp
        return f'{timestamp}.{record_msecs[:3]}.{record_msecs[3:]}'

    def format(self, record):
        """
        Apply log format with specified pattern.
        :param record: Format pattern.
        :return: formatted log content according to format pattern.
        """
        if record.pathname.startswith(INFER_INSTALL_PATH):
            # Get the relative path
            record.filepath = record.pathname[len(INFER_INSTALL_PATH):]
        elif "/" in record.pathname:
            record.filepath = record.pathname.strip().split("/")[-1]
        else:
            record.filepath = record.pathname
        record.sub_module = self.sub_module
        return super().format(record)


class RotatingLogFileHandler(RotatingFileHandler):
    def _open(self):
        return os.fdopen(os.open(self.baseFilename, os.O_RDWR | os.O_CREAT, 0o600), 'a')


def _filter_env_level():
    log_env_level = os.getenv(LOG_ENV, '1')
    if not isinstance(log_env_level, str) or not log_env_level.isdigit() \
            or int(log_env_level) < 0 or int(log_env_level) > 3:
        log_env_level = '1'
    return log_env_level


class LOGGER(logging.Logger):
    def __init__(self, logger_name, log_level=logging.WARNING):
        super(LOGGER, self).__init__(logger_name)
        self.model_name = logger_name
        self.data_formatter = DataFormatter(self.model_name, self._get_formatter())
        self.console_log_level = _name_to_log_level.get(
            _ms_level_to_name.get(_filter_env_level())) if log_level is None else log_level
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(level=self.console_log_level)
        console.setFormatter(self.data_formatter)
        self.addHandler(console)

    @staticmethod
    def _get_formatter():
        """

        :return: str, the string of log formatter.
        """
        formatter = '[%(levelname)s] %(sub_module)s(%(process)d:' \
                    '%(thread)d,%(processName)s):%(asctime)s ' \
                    '[%(filepath)s:%(lineno)d] %(message)s'
        return formatter

    def info(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.INFO) and os.getenv('RANK_ID', '0') == '0':
            self._log(logging.INFO, msg, args, **kwargs)

    def debug(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.DEBUG) and os.getenv('RANK_ID', '0') == '0':
            self._log(logging.DEBUG, msg, args, **kwargs)

    def warning(self, msg, *args, **kwargs):
        if self.isEnabledFor(logging.WARNING) and os.getenv('RANK_ID', '0') == '0':
            self._log(logging.WARNING, msg, args, **kwargs)

    def error(self, msg, *args, **kwargs):
        rank_id = os.getenv('RANK_ID', None)
        if rank_id and rank_id.isdigit() and 0 <= int(rank_id) < 8:
            msg = f"[The error from this card id ({rank_id})] " + msg
        if self.isEnabledFor(logging.ERROR):
            self._log(logging.ERROR, msg, args, **kwargs)

    def setup_logging_file(self, log_dir, max_size=100 * 1024 * 1024, backup_cnt=10):
        """Setup logging file."""
        if max_size > 1024 * 1024 * 1024 or max_size < 0:
            logging.error('single log file size should more than 0, less than or equal to 1G.')
            raise Exception('single log file size should more than 0, less than or equal to 1G.')
        if backup_cnt > 100 or backup_cnt < 0:
            logging.error('log file backup count should more than 0, less than or equal to 100')
            raise Exception('log file backup count should more than 0, less than or equal to 100')
        log_dir = os.path.realpath(log_dir)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir, mode=0o750)
        log_file_name = f"{self.model_name}.log"
        log_fn = os.path.join(log_dir, log_file_name)
        fh = RotatingLogFileHandler(log_fn, 'a', max_size, backup_cnt)
        fh.setFormatter(self.data_formatter)
        fh.setLevel(logging.INFO)
        self.addHandler(fh)

    def filter_log_str(self, msg) -> str:
        def _check_str(need_check_str):
            if len(need_check_str) > 10000:
                self.warning(f"Input should be <= 10000")
                return False
            filter_strs = ["\r", "\n", "\\r", "\\n"]
            for filter_str in filter_strs:
                if filter_str in need_check_str:
                    self.warning(f"Input should not be included \\r or \\n")
                    return False
            return True

        if isinstance(msg, str) and not _check_str(msg):
            return ''
        else:
            return msg

    def save_args(self, args):
        """
        :param args: input args param, just support argparse or dict
        :return: None
        """
        if isinstance(args, argparse.Namespace):
            args = vars(args)
        elif isinstance(args, dict):
            pass
        else:
            logging.error('This api just support argparse or dict, please check your input type.')
            raise Exception('This api just support argparse or dict, please check your input type.')
        self.info('Args:')
        args_copy = args.copy()
        for key, value in args_copy.items():
            self.info('--> %s: %s', key, self.filter_log_str(args_copy[key]))
        self.info('Finish read param')


class SingletonType(type):
    _instance_lock = threading.Lock()

    def __call__(cls, *args, **kwargs):
        if not hasattr(cls, "_instance"):
            with SingletonType._instance_lock:
                if not hasattr(cls, "_instance"):
                    cls._instance = super(SingletonType, cls).__call__(*args, **kwargs)
        return cls._instance


class LoggerSystem(metaclass=SingletonType):
    def __init__(self, model_name=LOG_TYPE, max_size=MAX_BYTES, backup_cnt=BACKUP_COUNT):
        self.model_name = model_name
        self.max_bytes = max_size
        self.backup_count = backup_cnt
        self.logger = None

    def init_logger(self, show_info_log=False, save_path=None):
        self.logger = LOGGER(self.model_name, logging.INFO if show_info_log else logging.WARNING)
        if save_path:
            self.logger.setup_logging_file(save_path, self.max_bytes, self.backup_count)

    def __getattr__(self, item):
        return object.__getattribute__(self.logger, item)


logger_instance = LoggerSystem(LOG_TYPE)
