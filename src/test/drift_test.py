import pytest
import numpy as np
import pandas as pd
import sys
from pathlib import Path

from src.monitoring.drift import emperical_distribution_function , feature_drift_detection

def test_sort():
    PROJECT_ROOT = Path().resolve().parents[0]
    sys.path.append(str(PROJECT_ROOT))
    data = np.array([1 , 4 , 2 , 9 , 3 , 4 , 1])
    F_nx = emperical_distribution_function(4 , data)
    assert F_nx == (6/7)



