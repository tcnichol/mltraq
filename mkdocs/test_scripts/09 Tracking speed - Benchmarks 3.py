#!/usr/bin/env python
# coding: utf-8

# # Tracking speed - Benchmarks 3
# 
# In this experiment, we evaluate how much time it takes to import a tracking package.
# we evaluate the tracking performance of:
# 
# * MLflow - https://mlflow.org/
# * WandB - https://wandb.ai/
# * Neptune - https://neptune.ai/
# * Aim - https://aimstack.io/
# * Comet - https://www.comet.com/
# * MLtraq - https://mltraq.com/
# 

# ## Experiment

# In[1]:


# Imports
import timeit

import pandas as pd


# In[2]:


# Test procedures


def test_MLflow():
    import mlflow

    return mlflow.__version__


def test_Neptune():
    import neptune

    return neptune.__version__


def test_WandB():
    import wandb

    return wandb.__version__


def test_Aim():
    import aim

    return aim.__version__.__version__


def test_Comet():
    import comet_ml

    return comet_ml.__version__


def test_MLtraq():
    import mltraq

    return mltraq.__version__


# In[3]:


# Running the test
times = {}

for method in ["Aim", "Comet", "WandB", "Neptune", "MLtraq", "MLflow"]:
    times[method] = timeit.timeit(f"test_{method}()", number=1, globals=globals())


# In[4]:


df = pd.Series(times, name="duration").sort_values().to_frame()
df["ratio_to_best"] = df["duration"] / df["duration"].iloc[0]
df


# ## Conclusion
# 
# MLtraq takes only 0.03 seconds.
# MLflow requires the highest time with 0.2 seconds.
# It seems a high variation, anyways, nothing significant motivating further analyses.

# In[ ]:




