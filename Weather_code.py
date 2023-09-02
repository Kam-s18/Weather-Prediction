from sklearn.datasets import load_boston
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import accuracy_score
import statsmodels.api as sm
plt.rcParams.update({'text.color' : "white", 'axes.labelcolor' : "white", 'xtick.color' : "white", 'ytick.color' : "white"})