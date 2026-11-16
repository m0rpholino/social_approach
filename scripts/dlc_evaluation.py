import sys
from pathlib import Path
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

import config
import deeplabcut as dlc

dlc.evaluate_network(config.DLC['config_path'])