import logging
import logging.config
import os

import yaml

from app.settings import settings

logger = logging.getLogger(__name__)


def config_logging() -> None:
    logger_config_path = f".config/logging_{settings.environment}.yaml"
    if not os.path.isfile(logger_config_path):
        logger.warning("Logger config %s is not found", logger_config_path)
        return

    with open(logger_config_path, encoding="utf-8") as f:
        config = yaml.safe_load(f.read())

    logging.config.dictConfig(config)
