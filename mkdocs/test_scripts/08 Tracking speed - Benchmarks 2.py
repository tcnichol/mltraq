#!/usr/bin/env python
# coding: utf-8

# # Tracking speed - Benchmarks 2
# 
# In this experiment, we extend the results of the Benchmarks notebook by assessing the performance to track 1D `NumPy` arrays.
# we evaluate the tracking performance of:
# 
# * MLflow - https://mlflow.org/
# * FastTrackML - https://github.com/G-Research/fasttrackml
# * WandB - https://wandb.ai/
# * Neptune - https://neptune.ai/
# * Aim - https://aimstack.io/
# * Comet - https://www.comet.com/
# * MLtraq - https://mltraq.com/
# 
# Varying:
# 
# * Number of arrays
# * Size of arrays
# 
# Configuration:
# * Tracking `Numpy` arrays as values, disabling everything else such as git, code, environment and system stats
# * Experiments running/tracking offline, logging disabled, storage on local filesystem
# * Every experiment starts with an empty directory for storage
# * Results averaged on `10` runs
# 

# ## Imports and utility functions

# In[1]:


get_ipython().run_line_magic('load_ext', 'pyinstrument')


# In[2]:


import logging
import shutil
import threading
import uuid
from contextlib import contextmanager, redirect_stderr, redirect_stdout
from os import devnull, environ, makedirs, remove

import aim
import comet_ml
import fasttrackml
import matplotlib.pyplot as plt
import mlflow
import neptune
import numpy as np
import pandas as pd
import wandb

import mltraq
from mltraq.utils.plot import bar_plot


# In[3]:


# Versions

print("mlflow", mlflow.__version__)
print("neptune", neptune.__version__)
print("wandb", wandb.__version__)
print("aim", aim.__version__.__version__)
print("comet", comet_ml.__version__)
print("mltraq", mltraq.__version__)
print("fasttrackml", "0.5.0b2")


# In[4]:


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


# ## Test procedure for WandB

# In[62]:


def test_wandb(n_values=1, size=1):
    """
    Test Weights & Biases tracking with a specified number of arrays.
    """

    # Required to silence Python output and disable sentry tracking.
    # This must be inside the test function to propagate to child processes.
    environ["WANDB_SILENT"] = "true"
    environ["WANDB_ERROR_REPORTING"] = "false"
    environ["WANDB_DISABLE_GIT"] = "true"
    environ["DISABLE_CODE"] = "true"

    tmp_dir = f"tmp/{uuid.uuid4()}"
    create_dir(f"{tmp_dir}/wandb")
    wandb.init(
        project=str(uuid.uuid4()),
        group=str(uuid.uuid4()),
        dir=tmp_dir,
        mode="offline",
    )
    for _ in range(n_values):
        table = wandb.Table(columns=["values"])
        table.add_data(np.zeros(size))
        wandb.log({"table": table})
        # Data is stored as JSON in files named table_something.table.json
    wandb.finish()


# In[65]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_wandb()\n')


# ## Test procedure for MLflow

# In[7]:


def test_mlflow(n_values=1, size=1):
    """
    Test MLflow tracking with a specified number of arrays.
    """

    create_dir("tmp")
    db_fname = f"tmp/{uuid.uuid4()}.db"
    mlflow.set_tracking_uri(f"sqlite:///{db_fname}")
    experiment_id = mlflow.create_experiment(str(uuid.uuid4()))
    with mlflow.start_run(experiment_id=experiment_id):
        for value_id in range(0, n_values):
            # Datasets don't seem to work, this doesn't result in any tracked data:
            # dataset = mlflow.data.from_numpy(np.zeros(size), source="test") # noqa
            # mlflow.log_input(dataset, context="training")# noqa

            # .log_param has a size limit of 6K
            # .log_metric works only with floats
            # .log_text writes to a file in dir ./mlruns

            # Not optimal, but there doesn't seem to be a better way without first
            # writing it to disk before creating one more copy.
            # large blobs are tacked on the filesystem as artifacts.
            # E.g., ".../artifacts/value_0".

            mlflow.log_text(str(np.zeros(size).tobytes()), f"value_{value_id}")

    mlflow.end_run()
    remove_file(db_fname)


# In[8]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_mlflow()\n')


# ## Test procedure for FastTrackML

# In[9]:


# Requires `fml server --log-level error`
# Data store in the "./artifacts" directory.


def test_fasttrackml(n_values=1, size=1):
    """
    Test FastTrackML tracking with a specified number of arrays.
    """

    fasttrackml.set_tracking_uri("http://localhost:5000")
    client = fasttrackml.FasttrackmlClient()
    experiment_id = "0"
    run = client.create_run(experiment_id)
    run_id = run.info.run_id
    for value_id in range(0, n_values):
        client.log_text(run_id, str(np.zeros(size).tobytes()), f"value_{value_id}")
    client.set_terminated(run_id)


# In[10]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_fasttrackml()\n')


# ## Test procedure for Neptune

# In[59]:


def test_neptune(n_values=1, size=1):
    """
    Test Neptune tracking with a specified number of arrays.
    """

    run = neptune.init_run(
        project=f"workspace/{str(uuid.uuid4())}",
        mode="offline",
        git_ref=False,
    )
    for _ in range(0, n_values):
        # https://docs.neptune.ai/logging/arrays_and_tensors/#logging-numpy-arrays
        run["totrack"].append(neptune.types.File.as_image(np.zeros((size, 1))))
        # File encoded in string format, example:
        # ./data-1.log:{"obj": {"type": "LogImages", "path": ["valuetotrack"],
        #               "values": [{"value": {"data": "iVBORw0KGgo ...
    run.wait()
    run.stop()


# In[61]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_neptune()\n')


# ## Test procedure for Aim

# In[13]:


def test_aim(n_values=1, size=1):
    """
    Test Aim tracking with a specified number of arrays.
    """

    create_dir("tmp/aim/")
    repo = f"tmp/aim/{uuid.uuid4()}"
    # Doc: https://aimstack.readthedocs.io/en/latest/refs/sdk.html#aim.sdk.run.Run
    # Experiments in Aim match to what we call runs in this notebook.
    run = aim.Run(
        repo=repo,
        experiment=str(uuid.uuid4()),
        system_tracking_interval=None,
        capture_terminal_logs=False,
    )
    for _ in range(0, n_values):
        # With:
        #    run.track({"value": [np.zeros(size)]}) # noqa
        # Error:
        #    TypeError: Unhandled non-native value `[0. 0. 0. 0. 0. 0. 0. 0. 0. 0.]`
        #    of type `<class 'numpy.ndarray'>`.
        run.track({"value": [np.zeros(size).tobytes()]})


# In[14]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_aim()\n')


# ## Test procedure for Comet

# In[39]:


def test_comet(n_values=1, size=1):
    """
    Test Comet tracking with a specified number of arrays.
    """

    tmp_dir = f"tmp/comet/{uuid.uuid4()}"
    create_dir(tmp_dir)
    run = comet_ml.OfflineExperiment(
        project_name=str(uuid.uuid4()),
        log_code=False,
        log_graph=False,
        log_env_gpu=False,
        log_env_cpu=False,
        log_env_network=False,
        log_env_disk=False,
        log_env_host=False,
        log_git_metadata=False,
        offline_directory=tmp_dir,
        experiment_key=str(uuid.uuid4().hex),
        display_summary_level=0,
    )
    for _ in range(0, n_values):
        # .log_other, .log_metrics don't really track
        # the value, it converts the object to a
        # short string eg. "[0, 0, ...., 0, 0]".
        run.log_text(str(np.zeros(size).tolist()))

    run.end()


# In[40]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_comet()\n')


# ## Test procedure for MLtraq

# In[17]:


def test_mltraq(n_values=1, size=1):
    """
    Test MLtraq tracking with a specified number of arrays.
    """

    create_dir("tmp")
    db_fname = f"tmp/{uuid.uuid4()}.db"
    session = mltraq.create_session(f"sqlite:///{db_fname}")
    with mltraq.options().ctx({"tqdm.disable": True}):
        experiment = session.create_experiment()
        with experiment.run() as run:
            run.fields.ds = mltraq.DataStore()
            run.fields.ds.value = []
            for _ in range(0, n_values):
                run.fields.ds.value.append(np.zeros(size))
        experiment.persist()
    experiment.delete()
    remove_file(db_fname)


# In[18]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_mltraq()\n')


# ## Defining the experiment

# In[19]:


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
            test_mlflow(n_values=run.params.n_values, size=run.params.size)
        elif run.params.method == "MLtraq":
            test_mltraq(n_values=run.params.n_values, size=run.params.size)
        elif run.params.method == "Neptune":
            test_neptune(n_values=run.params.n_values, size=run.params.size)
        elif run.params.method == "WandB":
            test_wandb(n_values=run.params.n_values, size=run.params.size)
        elif run.params.method == "Aim":
            test_aim(n_values=run.params.n_values, size=run.params.size)
        elif run.params.method == "Comet":
            test_comet(n_values=run.params.n_values, size=run.params.size)
        elif run.params.method == "FastTrackML":
            test_fasttrackml(n_values=run.params.n_values, size=run.params.size)
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
    shutil.rmtree("artifacts", ignore_errors=True)


# ## Defining the plots

# In[20]:


def report_results(experiment: mltraq.Experiment, save_svg_to=None):
    """
    Given an executed experiment, report the results with a plot and a table.
    """

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=[10, 2], nrows=1, ncols=3)
    fig.tight_layout()
    plt.subplots_adjust(hspace=0.4)

    with mltraq.options().ctx(
        {
            "matplotlib.rc": {"hatch.color": "white"},
            "matplotlib.style": "tableau-colorblind10",
        }
    ):

        bar_plot(
            experiment.runs.df(),
            x="n_values",
            x_label="Number of arrays",
            y="duration",
            group="method",
            yerr=True,
            ax=ax1,
            y_label="Duration (s)",
            y_lim={"bottom": 0},
            y_grid=False,
            legend={
                "loc": "lower center",
                "bbox_to_anchor": (0.5, -1.1),
                "fancybox": True,
                "shadow": False,
                "ncol": 2,
            },
        )

        bar_plot(
            experiment.runs.df(),
            x="size",
            x_label="Size of arrays",
            y="duration",
            yerr=True,
            y_label="Duration (s)",
            group="method",
            ax=ax2,
            y_lim={"bottom": 0},
            legend={
                "loc": "lower center",
                "bbox_to_anchor": (0.5, -1.1),
                "fancybox": True,
                "shadow": False,
                "ncol": 2,
            },
        )
        ax2.yaxis.label.set_visible(False)

        bar_plot(
            experiment.runs.df(),
            x="method",
            x_label="Method",
            y="duration",
            yerr=True,
            ax=ax3,
            y_lim={"bottom": 0},
            legend={
                "loc": "lower center",
                "bbox_to_anchor": (0.5, -1.1),
                "fancybox": True,
                "shadow": False,
                "ncol": 2,
            },
        )
        ax3.yaxis.label.set_visible(False)
        ax3.tick_params(axis="x", labelrotation=45)

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

# In[22]:


# Create an MLtraq session to track the benchmarks
session = mltraq.create_session("sqlite:///local/benchmarks2.db")


# ### Experiment 1: How long does tracking a 1D NumPy array of length 1M take?
# 
# In this experiment, we evaluate the time required to start a new experiment and track a single array of length 1M (default dtype is `numpy.float64`), which is 8M bytes if written to disk.
# 
# Most methods do NOT support serialization of NumPy arrays, and our test procedures take care of doing it.

# In[32]:


len(np.zeros(10**6).tobytes())


# In[46]:


e = session.create_experiment("exp-1", if_exists="replace")
e.add_runs(
    method=[
        "Aim",
        "Comet",
        "WandB",
        "Neptune",
        "MLtraq",
        "MLflow",
        "FastTrackML",
    ],
    i=range(10),
    n_values=[1],
    size=[int(10**6)],
)

# Parallelization is disabled as it might affect results.
e.execute([cleanup, eval_time, cleanup], n_jobs=1).persist(if_exists="replace")


# In[47]:


# Report results
report_results(session.load("exp-1"))


# * Results are very similar to experiment 1 in the Benchmarks notebook. The time cost is dominated by start up (database/filesystem, threads).
# * MLtraq is faster than Neptune due to the inefficient serialization strategy of Neptune, that relies on a string format to encode the contents of the array.

# ### Experiment 2: How much time to track 1, 5, 10 arrays of size 10K, 100K and 1M?

# In[23]:


e = session.create_experiment("exp-2", if_exists="replace")
e.add_runs(
    method=[
        "Aim",
        "Comet",
        "Neptune",
        "MLtraq",
        "MLflow",
        "FastTrackML",
        "WandB",
    ],
    i=range(10),
    n_values=[1, 5, 10],
    size=[int(10**4), int(10**5), int(10**6)],
)

# WandB often fails with errors "ValueError: I/O operation on closed file.",
# and "BrokenPipeError: [Errno 32] Broken pipe".

# Parallelization is disabled as it might affect results.
e.execute([cleanup, eval_time, cleanup], n_jobs=1).persist(if_exists="replace")


# In[24]:


# Report results
report_results(session.load("exp-2"))


# Trends remain similar to Experiment 1, except WandB whose time explodes.

# ### Experiment 3: How much time to track 1, 10, 100, 1K and 10K arrays of size 1K?

# In[32]:


e = session.create_experiment("exp-3", if_exists="replace")
e.add_runs(
    method=["Neptune", "MLtraq", "Comet", "FastTrackML"],
    i=range(10),
    n_values=[1, int(10**1), int(10**2), int(10**3), int(10**4)],
    size=[1, int(10**2), int(10**3)],
)

# Parallelization is disabled as it might affect results.
e.execute([cleanup, eval_time, cleanup], n_jobs=1).persist(if_exists="replace")


# In[38]:


# Report results
report_results(session.load("exp-3"))


# * Comet and FastTrackML are the worst performing, with time dominated by creating the many files.
# * MLtraq is 5x faster than Neptune and 20x faster than the others. Its strategy to create a bag of objects that are stored together is much more efficient than storing each object independently, especially if the number of objects is high.

# ### Aggregates

# In[41]:


dfs = [session.load(f"exp-{idx}").runs.df() for idx in [1, 2, 3]]
df = pd.concat(dfs).groupby("method").duration.mean().sort_values().to_frame()
df["ratio_to_best"] = df["duration"] / df["duration"].iloc[0]
df


# ### Save plots to SVG files

# In[42]:


# Create an MLtraq session to reload the benchmarks
session = mltraq.create_session("sqlite:///local/benchmarks2.db")

# Save plots to SVG files
report_results(
    session.load("exp-1"),
    save_svg_to="../mkdocs/assets/img/benchmarks2/exp-1.svg",
)
report_results(
    session.load("exp-2"),
    save_svg_to="../mkdocs/assets/img/benchmarks2/exp-2.svg",
)
report_results(
    session.load("exp-3"),
    save_svg_to="../mkdocs/assets/img/benchmarks2/exp-3.svg",
)


# ## Conclusion
# 
# In the experiments, MLtraq is, on average, 4-100x faster than the other methods. As the number and size of the tracked arrays increases, MLtraq stands out even more, being 20x faster than others. MLtraq has been designed to have a robust and complete serialization/storage mechanism for complex Python objects. If tracking a wide range of objects and speed are your priorities, you should consider MLtraq to run your experiments.

# In[ ]:




