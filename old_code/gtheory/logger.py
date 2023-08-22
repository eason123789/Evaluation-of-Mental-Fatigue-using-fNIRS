
# https://stackoverflow.com/a/44175370/6446053
import logging
import sys
logging.getLogger().setLevel(logging.INFO)
gettrace = getattr(sys, 'gettrace', None)


logging.basicConfig(
    format='%(asctime)s %(levelname)-8s %(message)s',
    level=logging.INFO,
    datefmt='%Y-%m-%d %H:%M:%S')

logging.info('Complete adjusting')