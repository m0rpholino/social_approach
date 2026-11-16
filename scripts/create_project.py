from pathlib import Path
import sys
src_path = Path(__file__).parent.parent / 'src'
sys.path.append(str(src_path))

import config
import utils

utils.create_project(config.PROJECT_ROOT)
print(f'Project created at: {config.PROJECT_ROOT}')