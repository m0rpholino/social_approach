# labels module

from typing import Optional, Union, Tuple
from pathlib import Path
import pandas as pd
import numpy as np

def convert_to_binary(labels: Union[str, Path, pd.DataFrame],
                      behavior: str,
                      slice: Optional[Tuple[int, int]] = None):
    if isinstance(labels, str) or isinstance(labels, Path):
        labels_df = pd.read_csv(labels)
        events = labels_df.copy()
    elif isinstance(labels, pd.DataFrame):
        events = labels_df.copy()
    else:
        raise TypeError(f'Unsupported type for labels: {type(labels)}')
    
    events = events.replace(behavior, 1)
    
    if slice:
        duration = slice[1] - slice[0]
    else:
        duration = int(events['Observation duration'][0] * events['FPS'][0])
    
    binary_labels = np.zeros(duration, dtype=np.uint8)

    for i in range(0, len(events), 2):
        binary_labels[events.loc[i, 'Image index']:events.loc[i+1, 'Image index']] = events.loc[i, 'Behavior']
    
    return binary_labels

