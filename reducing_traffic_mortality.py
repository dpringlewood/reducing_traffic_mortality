"""
This is a project to find patterns in road traffic accidents
so that we know where to focus our attention to bring down
the accident rate.

Can we group states into similar profiles.
"""

import pandas as pd
import matplotlib as plt
import seaborn as sns
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA