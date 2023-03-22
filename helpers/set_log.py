import logging
import yaml
import logging.config
import os


def setup_logging(default_path='./Log/log.yaml', default_level=logging.INFO):
    path = default_path
    if os.path.exists(path):
        with open(path, 'r', encoding='utf-8') as f:
            log_config = yaml.load(f)
            logging.config.dictConfig(log_config)
    else:
        logging.basicConfig(level=default_level)
