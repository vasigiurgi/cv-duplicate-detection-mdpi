import logging
import sys

logger = logging.getLogger("mdpi_assessment")
logger.setLevel(logging.DEBUG)

ch = logging.StreamHandler(sys.stdout)
ch.setLevel(logging.DEBUG)

formatter = logging.Formatter("[%(levelname)s] %(asctime)s - %(message)s")
ch.setFormatter(formatter)

logger.addHandler(ch)
