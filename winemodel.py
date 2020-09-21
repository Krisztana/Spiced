from sklearn.dummy import DummyClassifier
from sklearn.datasets import load_wine
import pandas as pd

def train_model(X, y):
    m = DummyClassifier()
    m.fit(X, y)
    return m
