#!/usr/bin/env python
# coding: utf-8

# # Tracking speed - Benchmarks
# 
# In this experiment, we evaluate the tracking performance of:
# 
# * MLflow - https://mlflow.org/
# * WandB - https://wandb.ai/
# * Neptune - https://neptune.ai/
# * Aim - https://aimstack.io/
# * Comet - https://www.comet.com/
# * MLtraq - https://mltraq.com/
# 
# Varying:
# 
# * Number of experiments tracked
# * Number of runs tracked
# * Number of values tracked
# 
# Configuration:
# * Tracking `float` values, disabling everything else such as git, code, environment and system stats
# * Experiments running/tracking offline, logging disabled, storage on local filesystem
# * Every experiment starts with an empty directory for storage
# * Results averaged on `10` runs
# 

# ## Imports and utility functions

# In[2]:


get_ipython().run_line_magic('load_ext', 'pyinstrument')


# In[3]:


import logging
import shutil
import threading
import uuid
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull, environ, makedirs, remove

import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import wandb

import mltraq
from mltraq.utils.plot import bar_plot


# In[4]:


# Versions

print("mlflow", mlflow.__version__)
print("wandb", wandb.__version__)
print("mltraq", mltraq.__version__)


# In[5]:


# Utility functions


@contextmanager
def suppress_stdout_stderr():
    """
    A context manager that redirects stdout and stderr to devnull.
    """
    with open(devnull, "w") as fnull:
        with redirect_stderr(fnull) as err, redirect_stdout(fnull) as out:
            yield (err, out)


def remove_file(pathname):
    """
    Remove file `pathname` if existing.
    """
    try:
        remove(pathname)
    except OSError:
        pass


def create_dir(pathdir):
    """
    Create `pathdir` recursively. If it already exists, do nothing.
    """
    makedirs(pathdir, exist_ok=True)


# In[ ]:





# ## Test procedure for WandB

# In[ ]:





# In[29]:


def test_wandb(n_experiments=1, n_runs=1, n_values=1):
    """
    Test Weights & Biases tracking with a specified number of experiments, runs and values.
    """

    # Required to silence Python output and disable sentry tracking.
    # This must be inside the test function to propagate to child processes.
    environ["WANDB_SILENT"] = "true"
    environ["WANDB_ERROR_REPORTING"] = "false"
    environ["WANDB_DISABLE_GIT"] = "true"
    environ["DISABLE_CODE"] = "true"

    tmp_dir = f"tmp/{uuid.uuid4()}"
    create_dir(f"{tmp_dir}/wandb")
    for _ in range(n_experiments):
        experiment_id = str(uuid.uuid4())
        for _ in range(n_runs):
            run_id = str(uuid.uuid4())
            wandb.init(
                project=experiment_id,
                group=run_id,
                dir=tmp_dir,
                mode="offline",
            )
            for _ in range(n_values):
                wandb.log({"value": 123.45})
    wandb.finish()


# In[30]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_wandb()\n')


# ## Test procedure for MLflow

# In[31]:


def test_mlflow(n_experiments=1, n_runs=1, n_values=1):
    """
    Test MLflow tracking with a specified number of experiments, runs and values.
    """

    create_dir("tmp")
    db_fname = f"tmp/{uuid.uuid4()}.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_fname}")
    for _ in range(n_experiments):
        experiment_id = mlflow.create_experiment(str(uuid.uuid4()))
        for _ in range(n_runs):
            with mlflow.start_run(experiment_id=experiment_id):
                for _ in range(0, n_values):
                    mlflow.log_metric(key="value", value=123.45)
            mlflow.end_run()
    remove_file(db_fname)


# In[32]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_mlflow()\n')


# ## Test procedure for Neptune

# In[33]:





# In[34]:





# ## Test procedure for Aim

# 

# In[35]:





# In[36]:





# ## Test procedure for Comet

# In[37]:





# In[38]:





# ## Test procedure for MLtraq

# In[39]:


def test_mltraq(n_experiments=1, n_runs=1, n_values=1):
    """
    Test MLtraq tracking with a specified number of experiments, runs and values.
    """

    create_dir("tmp")
    db_fname = f"tmp/{uuid.uuid4()}.db"
    session = mltraq.create_session(f"sqlite:///{db_fname}")
    with mltraq.options().ctx({"tqdm.disable": True}):
        for _ in range(n_experiments):
            experiment = session.create_experiment()
            for _ in range(n_runs):
                with experiment.run() as run:
                    run.fields.value = []
                    for _ in range(0, n_values):
                        run.fields.value.append(123.45)
            experiment.persist()
    remove_file(db_fname)


# In[40]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_mltraq()\n')


# ## Defining the experiment

# In[7]:


def eval_time(run: mltraq.Run):
    """
    Measure the time required to track a set of experiments.
    """

    # Disable logging
    logging.disable()

    with suppress_stdout_stderr():
        log = mltraq.Sequence()

        # Start tracking time.
        log.append(tag="begin")
        if run.params.method == "MLflow":
            test_mlflow(
                n_experiments=run.params.n_experiments,
                n_runs=run.params.n_runs,
                n_values=run.params.n_values,
            )
        elif run.params.method == "MLtraq":
            test_mltraq(
                n_experiments=run.params.n_experiments,
                n_runs=run.params.n_runs,
                n_values=run.params.n_values,
            )
        elif run.params.method == "WandB":
            test_wandb(
                n_experiments=run.params.n_experiments,
                n_runs=run.params.n_runs,
                n_values=run.params.n_values,
            )
        else:
            raise Exception("unknown method")

        log.append(tag="end")
        durations = log.df().pivot_table(index="tag", values="timestamp")["timestamp"]
        run.fields.duration = (durations.end - durations.begin).total_seconds()
        run.fields.n_threads = threading.active_count()
        run.fields |= run.params


def cleanup(run: mltraq.Run):
    # Remove temporary files
    shutil.rmtree(".neptune", ignore_errors=True)
    shutil.rmtree(".cometml-runs", ignore_errors=True)
    shutil.rmtree("mlruns", ignore_errors=True)
    shutil.rmtree("tmp", ignore_errors=True)


def report_results(experiment: mltraq.Experiment, save_svg_to=None):
    """
    Given an executed experiment, report the results with a plot and a table.
    """

    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(figsize=[6, 5], nrows=2, ncols=2)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.4)

    with mltraq.options().ctx({"matplotlib.style": "seaborn-v0_8-pastel"}):

        bar_plot(
            experiment.runs.df(),
            x="n_experiments",
            x_label="Number of experiments",
            y="duration",
            group="method",
            yerr=True,
            ax=ax1,
            y_label="Duration (s)",
            y_lim={"bottom": 0},
            y_grid=False,
        )

        bar_plot(
            experiment.runs.df(),
            x="method",
            x_label="Method",
            y="duration",
            yerr=True,
            ax=ax2,
            y_lim={"bottom": 0},
        )
        ax2.yaxis.label.set_visible(False)
        ax2.tick_params(axis="x", labelrotation=25)

        bar_plot(
            experiment.runs.df(),
            x="n_runs",
            x_label="Number of runs",
            y="duration",
            yerr=True,
            y_label="Duration (s)",
            group="method",
            ax=ax3,
            y_lim={"bottom": 0},
        )

        bar_plot(
            experiment.runs.df(),
            x="n_values",
            x_label="Number of values",
            y="duration",
            yerr=True,
            group="method",
            ax=ax4,
            y_label="duration (s)",
            y_lim={"bottom": 0},
        )
        ax4.yaxis.label.set_visible(False)

    if save_svg_to:
        plt.savefig(save_svg_to, bbox_inches="tight", pad_inches=0.1)

    plt.show()

    # Display aggregated results table (duration and multiplier to the best-performing method)
    print("\nAveraged results by method\n")
    df = experiment.runs.df().groupby("method")["duration"].mean().sort_values().to_frame()
    df["ratio_to_best"] = df["duration"] / df["duration"].iloc[0]
    display(df)
    print("\n")


# ## Experiments
# 
# * In the rest of the notebook, we experiment varying number of experiments, runs and values.
# * With some differences, all methods capture these parameters.
# 

# In[20]:


# Create an MLtraq session to track the benchmarks
session = mltraq.create_session("sqlite:///local/benchmarks.db")


# ### Experiment 1: How long does tracking a single value take?
# 
# In this experiment, we evaluate the time required to start a new experiment and track a single value.
# This experiment lets us compare the start-up time of the methods, regardless of how many values we track.

# In[21]:


e = session.create_experiment("exp-1", if_exists="replace")
e.add_runs(
    method=["WandB", "MLtraq", "MLflow"],
    i=range(10),
    n_experiments=[1],
    n_runs=[1],
    n_values=[1],
)

# Parallelization is disabled as it might affect results.
e.execute([cleanup, eval_time, cleanup], n_jobs=1).persist(if_exists="replace")

# Report results
report_results(e)


# The analysis of the test procedures with `pyinstrument` reveals where most of the time is spent for each method:
# 
# * WandB and MLflow are the worst performing, with time dominated by threading and events management.
# * Aim follows by spending most of the time creating and managing the database.
# * Comet is next, with threading management taking most of the time.
# * MLtraq spends most of the time writing to SQLite, with no threading.
# * Comet is the best performing, with no threading, no SQLite database, simply writing the tracking data to disk.
# 
# In summary, the less you do to start up, the faster you are. Threading and communication are expensive, as well as database management.

# ### Experiment 2: How much time to track 1K and 10K values?

# In[22]:


e = session.create_experiment("exp-2")
e.add_runs(
    method=["WandB","MLtraq", "MLflow"],
    i=range(10),
    n_experiments=[1],
    n_runs=[1],
    n_values=[10**3],
)

# Parallelization disabled as it might affect results.
e.execute([cleanup, eval_time, cleanup], n_jobs=1).persist(if_exists="replace")

# Report results
report_results(e)


# In[23]:


e = session.create_experiment("exp-2a")
e.add_runs(
    method=["WandB", "MLtraq", "MLflow"],
    i=range(10),
    n_experiments=[1],
    n_runs=[1],
    n_values=[10**4],
)

# Parallelization disabled as it might affect results.
e.execute([cleanup, eval_time, cleanup], n_jobs=1).persist(if_exists="replace")

# Report results
report_results(e)


# * Performance changes dramatically, with WandB, MLflow and Aim being the worst performing, either due to thread communication (WandB, Aim) or database management (MLflow).
# * Comet is next, followed by Neptune and MLtraq.
# 
# The advantage of MLtraq is in how the data is tracked and stored. Being very close to simply adding an element to an array and serializing it to an SQLite column value with the speedy DATAPAK serialization strategy, it is hard to beat.

# ### Experiment 3: How much time to track 10 runs?

# In[24]:


e = session.create_experiment("exp-3")
e.add_runs(
    method=["WandB", "MLtraq", "MLflow"],
    i=range(10),
    n_experiments=[1],
    n_runs=[10],
    n_values=[1],
)

# Parallelization is disabled as it might affect results.
e.execute([cleanup, eval_time, cleanup], n_jobs=1).persist(if_exists="replace")

# Report results
report_results(e)


# The results are very similar to Experiment 1, with MLtraq now performing at the top.
# 
# * Neptune and Comet use the filesystem as a database, and MLtraq uses SQLite. As the number of files written by Neptune/Comet increases, it becomes more expensive than writing efficiently to a single database file. Writing to a single SQLite file can be 35% faster than writing to many filesystem files. See https://www.sqlite.org/fasterthanfs.html for more details.
# 
# * Comet creates a ZIP of the files to be uploaded to their cloud, which results in an additional time penalty.
# 
# 

# ### Experiment 4: How much time to track 100 runs?

# In[25]:


e = session.create_experiment("exp-4")
e.add_runs(
    method=["MLtraq"],
    i=range(10),
    n_experiments=[1],
    n_runs=[100],
    n_values=[1],
)

# Parallelization is disabled as it might affect results.
e.execute([cleanup, eval_time, cleanup], n_jobs=1).persist(if_exists="replace")

# Report results
report_results(e)


# The results are very similar to Experiment 4. Creating a ZIP of the logs is very expensive for Comet. The other methods have been excluded as they are orders of magnitude slower.

# ### Experiment 5: How much time to track 1K runs and 1K values each?

# In[26]:





# * As we increase the number of tracked values or runs, MLtraq becomes more and more competitive. With no threading and no filesystem bottleneck, it is the fastest method for realistic workloads.
# * With up to 1K runs and values, MLtraq is 25x faster than Neptune on average.
# The other methods, WandB, Aim, Comet and MLflow, are orders of magnitude slower.

# ### Aggregates

# In[27]:


dfs = [session.load(f"exp-{idx}").runs.df() for idx in [1, 2, 3, 4, 5]]
df = pd.concat(dfs).groupby("method").duration.mean().sort_values().to_frame()
df["ratio_to_best"] = df["duration"] / df["duration"].iloc[0]
df


# ### Save plots to SVG files

# In[8]:


# Create an MLtraq session to reload the benchmarks
session = mltraq.create_session("sqlite:///local/benchmarks.db")

# Save plots to SVG files
report_results(
    session.load("exp-1"),
    save_svg_to="../mkdocs/assets/img/benchmarks/exp-1.svg",
)
report_results(
    session.load("exp-2"),
    save_svg_to="../mkdocs/assets/img/benchmarks/exp-2.svg",
)
report_results(
    session.load("exp-2a"),
    save_svg_to="../mkdocs/assets/img/benchmarks/exp-2a.svg",
)
report_results(
    session.load("exp-3"),
    save_svg_to="../mkdocs/assets/img/benchmarks/exp-3.svg",
)
report_results(
    session.load("exp-4"),
    save_svg_to="../mkdocs/assets/img/benchmarks/exp-4.svg",
)
report_results(
    session.load("exp-5"),
    save_svg_to="../mkdocs/assets/img/benchmarks/exp-5.svg",
)


# ## Conclusion
# 
# In the experiments, MLtraq is, on average, 7-30x faster than the other methods. The benefits of MLtraq against the most popular choices on workloads with hundreds of thousands of runs and tracked values are even more accentuated and closer to 100x. If speed is among your priorities, you should consider MLtraq to run your experiments. The primary disadvantage of MLtraq is its lack of streaming of the tracking data and the missing web dashboard.
# 

# In[ ]:




