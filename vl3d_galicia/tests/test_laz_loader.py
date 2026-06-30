import pytest
import numpy as np
import sys
import os

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.data.classes import map_labels

def test_map_labels():
    # ASPRS to our labels
    # 2: Ground -> 0
    # 3: Low Veg -> 1
    # 4: Mid Veg -> 2
    # 5: High Veg -> 3
    # 6: Building -> 4
    # 9: Water -> 5
    # Others -> 6 (Unreliable)
    
    classification = np.array([1, 2, 3, 4, 5, 6, 7, 9, 12, 17, 65])
    expected = np.array([6, 0, 1, 2, 3, 4, 6, 5, 6, 6, 6])
    
    result = map_labels(classification)
    
    np.testing.assert_array_equal(result, expected)

def test_missing_laz():
    # Placeholder for checking if the script handles missing laz files gracefully
    pass
