
import glob
import os
from gtheory.utils.misc import sort_path
from gtheory import config as cf
from glob import glob
pt = cf.root_path ()

path_epoch = sort_path ( glob ( f"{pt ['dir_root']}/*/*/*.fif" ) )
[os.remove ( filePath ) for filePath in path_epoch]
