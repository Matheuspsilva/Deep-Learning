import pandas as pd
import numpy as np
from keras.models import Sequential
from keras.layers import Dense, Dropout


previsores = pd.read_csv('entradas-breast.csv')
classe = pd.read_csv('saidas-breast.csv')
