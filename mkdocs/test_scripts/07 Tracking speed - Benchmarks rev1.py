#!/usr/bin/env python
# coding: utf-8

# # Tracking speed - Benchmarks
# 
# *Latest update: 2023.03.04*
# 
# 
# 
# In this experiment, we evaluate the tracking performance of:
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
# * Number of runs tracked
# * Number of values tracked
# 
# Configuration:
# * Keeping the number of experiments fixed to `1` (not all methods model both experiments and runs)
# * Tracking `float` values, disabling everything else such as git, code, environment and system stats
# * Experiments running/tracking offline, logging disabled, storage on local filesystem
# * Every experiment starts with an empty directory for storage
# * Results averaged on `10` runs
# 
# Comments:
# * FastTrackML requires a running server, the command used is `fml server --log-level error`. The results on "start-up time" for FastTrackML ignore the cost of starting up the server (which creates a SQLite file, schema compatible with MLflow).
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


import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import wandb

import mltraq
from mltraq.utils.plot import bar_plot


# In[3]:


# Versions

print("mlflow", mlflow.__version__)
print("wandb", wandb.__version__)
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

# In[5]:


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


# In[6]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_wandb()\n')


# ## Test procedure for MLflow

# In[7]:


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


# In[8]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_mlflow()\n')


# ## Test procedure for FastTrackML

# In[11]:


# requires `fml server --log-level error`
def test_fasttrackml(n_experiments=1, n_runs=1, n_values=1):
    """
    Test FastTrackML tracking with a specified number of experiments, runs and values.
    """

    fasttrackml.set_tracking_uri("http://localhost:5000")
    client = fasttrackml.FasttrackmlClient()
    for i in range(n_experiments):
        experiment_id = str(i)
        for _ in range(n_runs):
            run = client.create_run(experiment_id)
            run_id = run.info.run_id
            for _ in range(0, n_values):
                client.log_metric(run_id, "value", 123.45)
        client.set_terminated(run_id)


# In[12]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_fasttrackml()\n')


# ## Test procedure for Neptune

# In[13]:


def test_neptune(n_experiments=1, n_runs=1, n_values=1):
    """
    Test Neptune tracking with a specified number of experiments, runs and values.
    """

    for _ in range(n_experiments):
        # No "experiment" concept in Neptune, and we cannot create "projects" offline on the free plan.
        # Also, not easy to set a custom directory, it will log things in .neptune/
        for _ in range(n_runs):
            run = neptune.init_run(
                project=f"workspace/{str(uuid.uuid4())}",
                mode="offline",
                git_ref=False,
            )
            for _ in range(0, n_values):
                run["value"].append(123.45)
            run.wait()
            run.stop()


# In[14]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_neptune()\n')


# ## Test procedure for Aim

# In[15]:


def test_aim(n_experiments=1, n_runs=1, n_values=1):
    """
    Test Aim tracking with a specified number of experiments, runs and values.
    """

    create_dir("tmp/aim/")
    repo = f"tmp/aim/{uuid.uuid4()}"
    for _ in range(n_experiments):
        for _ in range(n_runs):
            run_id = str(uuid.uuid4())
            # Doc: https://aimstack.readthedocs.io/en/latest/refs/sdk.html#aim.sdk.run.Run
            # Experiments in Aim match to what we call runs in this notebook.
            run = aim.Run(
                repo=repo,
                experiment=run_id,
                system_tracking_interval=None,
                capture_terminal_logs=False,
            )
            for _ in range(0, n_values):
                run.track({"value": 123.45})


# In[16]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_aim()\n')


# ## Test procedure for Comet

# In[72]:


def test_comet(n_experiments=1, n_runs=1, n_values=1):
    """
    Test Comet tracking with a specified number of experiments, runs and values.
    """

    tmp_dir = f"tmp/comet/{uuid.uuid4()}"
    create_dir(tmp_dir)
    for _ in range(n_experiments):
        experiment_id = str(uuid.uuid4())
        for _ in range(n_runs):
            run_id = str(uuid.uuid4().hex)  # Only alphanum IDs for Comet experiments, which match our semantics of runs
            run = comet_ml.OfflineExperiment(
                project_name=experiment_id,
                log_code=False,
                log_graph=False,
                log_env_gpu=False,
                log_env_cpu=False,
                log_env_network=False,
                log_env_disk=False,
                log_env_host=False,
                log_git_metadata=False,
                offline_directory=tmp_dir,
                experiment_key=run_id,
                display_summary_level=0,
            )
            for _ in range(0, n_values):
                run.log_metrics({"value": 123.45})
            run.end()


# In[73]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_comet()\n')


# ## Test procedure for MLtraq

# In[19]:


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


# In[20]:


get_ipython().run_cell_magic('pyinstrument', '', 'test_mltraq()\n')


# ## Defining the experiment

# In[69]:


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
        if run.params.method == "FastTrackML":
            test_fasttrackml(
                n_experiments=run.params.n_experiments,
                n_runs=run.params.n_runs,
                n_values=run.params.n_values,
            )
        elif run.params.method == "MLflow":
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
        elif run.params.method == "Neptune":
            test_neptune(
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
        elif run.params.method == "Aim":
            test_aim(
                n_experiments=run.params.n_experiments,
                n_runs=run.params.n_runs,
                n_values=run.params.n_values,
            )
        elif run.params.method == "Comet":
            test_comet(
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


# ## Defining the plots

# In[71]:


def report_results(experiment: mltraq.Experiment, save_svg_to=None):
    """
    Given an executed experiment, report the results with a plot and a table.
    """

    fig, (ax1, ax2, ax3) = plt.subplots(figsize=[10, 2], nrows=1, ncols=3)
    fig.tight_layout()
    with mltraq.options().ctx(
        {
            "matplotlib.rc": {"hatch.color": "white"},
            "matplotlib.style": "tableau-colorblind10",
        }
    ):

        bar_plot(
            experiment.runs.df(),
            x="n_runs",
            x_label="Number of runs",
            y="duration",
            yerr=True,
            y_label="Duration (s)",
            group="method",
            ax=ax1,
            y_lim={"bottom": 0},
            hatches=False,
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
            x="n_values",
            x_label="Number of values",
            y="duration",
            yerr=True,
            group="method",
            ax=ax2,
            y_label="duration (s)",
            y_lim={"bottom": 0},
            hatches=False,
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

# In[23]:


# Create an MLtraq session to track the benchmarks
session = mltraq.create_session("sqlite:///local/benchmarks_rev1.db")


# ### Experiment 1: How long does tracking a single value take?
# 
# In this experiment, we evaluate the time required to start a new experiment and track a single value.
# This experiment lets us compare the start-up time of the methods, regardless of how many values we track.
# FastTrackML requires a running web server (executed with `fml server --log-level error`).

# In[28]:


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
    n_experiments=[1],
    n_runs=[1],
    n_values=[1, 10],
)

# Parallelization is disabled as it might affect results.
e.execute([cleanup, eval_time, cleanup], n_jobs=1).persist(if_exists="replace")


# In[47]:


# Report results
report_results(session.load("exp-1"))


# The analysis of the test procedures with `pyinstrument` reveals where most of the time is spent for each method:
# 
# * WandB and MLflow are the worst performing, with time dominated by threading, events/databases management.
# * Aim follows by spending most of the time creating and managing the database.
# * Comet is next, with threading management taking most of the time.
# * FastTrackML is remarkably fast to create new runs, offering API compatibility with MLFlow. It is fast also because it requires a server running in the background, eliminating most of the startup cost.
# * MLtraq spends most of the time writing to SQLite, with no threading. Again, database management is expensive.
# * Comet is the best performing, with no threading, no SQLite database, simply writing the tracking data to disk. The least you do, the fastest you do it.
# 
# In summary, the less you do to start up, the faster you are. Threading and communication are expensive, as well as database management.

# ### Experiment 2: How much time to track 100-100K values?

# In[36]:


e = session.create_experiment("exp-2")
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
    n_experiments=[1],
    n_runs=[1],
    n_values=[10**2, 10**3, 10**4],
)

# Parallelization disabled as it might affect results.
e.execute([cleanup, eval_time, cleanup], n_jobs=1).persist(if_exists="replace")


# In[37]:


# Report results
report_results(session.load("exp-2"))


# * Performance changes dramatically, with WandB, MLflow, FastTrackML and Aim being the worst performing, either due to thread communication (WandB, Aim) or database management (MLflow, FastTrackML). MLflow and FastTrackML have the same database schema, which is rather expensive to maintain.
# * Comet is next, followed by Neptune and MLtraq.
# 
# The advantage of MLtraq is in how the data is tracked and stored. Being very close to simply adding an element to an array and serializing it to an SQLite column value with the speedy DATAPAK serialization strategy, it is hard to beat.

# ### Experiment 3: How much time to track 10 runs?

# In[38]:


e = session.create_experiment("exp-3")
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
    n_experiments=[1],
    n_runs=[10],
    n_values=[1],
)

# Parallelization is disabled as it might affect results.
e.execute([cleanup, eval_time, cleanup], n_jobs=1).persist(if_exists="replace")


# In[39]:


# Report results
report_results(session.load("exp-3"))


# The results are very similar to Experiment 1, with MLtraq now performing at the top.
# 
# * WandB, MLflow, Aim are the worst performing (as in Experiment 1). FastTrackML is handling the creation of new runs much faster than MLflow. This is where it shines.
# * Neptune and Comet use the filesystem as a database, and MLtraq uses SQLite. As the number of files written by Neptune/Comet increases, it becomes more expensive than writing efficiently to a single database file. Writing to a single SQLite file can be 35% faster than writing to many filesystem files. See https://www.sqlite.org/fasterthanfs.html for more details.
# 
# * Comet creates a ZIP of the files to be uploaded to their cloud, which results in an additional time penalty.
# 
# 

# ### Experiment 4: How much time to track 100 runs?

# In[40]:


e = session.create_experiment("exp-4")
e.add_runs(
    method=["Comet", "Neptune", "MLtraq", "FastTrackML"],
    i=range(10),
    n_experiments=[1],
    n_runs=[100],
    n_values=[1],
)

# Parallelization is disabled as it might affect results.
e.execute([cleanup, eval_time, cleanup], n_jobs=1).persist(if_exists="replace")


# In[41]:


# Report results
report_results(session.load("exp-4"))


# The results are very similar to Experiment 4. Creating hundreds of files, and zipping them, is very expensive for Comet. The other methods (MLFlow, WandB) have been excluded as they are orders of magnitude slower. They're out of the game.

# ### Experiment 5: How much time to track 1K runs and 1K values each?

# In[ ]:


e = session.create_experiment("exp-5")
e.add_runs(
    method=["Neptune", "MLtraq"],
    i=range(10),
    n_experiments=[1],
    n_runs=[500, 1000],
    n_values=[500, 1000],
)

# Parallelization is disabled as it might affect results.
e.execute([cleanup, eval_time, cleanup], n_jobs=1).persist(if_exists="replace")


# In[49]:


# Report results
report_results(session.load("exp-5"))


# * As we increase the number of tracked values or runs, MLtraq becomes more and more competitive. With no threading and no filesystem bottleneck, it is the fastest method for realistic workloads.
# * With up to 1K runs and values, MLtraq is 23x faster than Neptune on average.
# The other methods, WandB, Aim, Comet, MLflow and FastTrackML are orders of magnitude slower.

# ### Aggregates

# In[50]:


# Warning: Aggregates on multiple experiments, not all methods used on all experiments.
dfs = [session.load(f"exp-{idx}").runs.df() for idx in [1, 2, 3, 4, 5]]
df = pd.concat(dfs).groupby("method").duration.mean().sort_values().to_frame()
df["ratio_to_best"] = df["duration"] / df["duration"].iloc[0]
df


# ### Save plots to SVG files

# In[51]:


# Create an MLtraq session to reload the benchmarks
session = mltraq.create_session("sqlite:///local/benchmarks_rev1.db")

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
# In the experiments, MLtraq is, on average, 5-25x faster than the other methods. The speedup of MLtraq against the most popular choices on workloads with hundreds of thousands of runs and tracked values are even more accentuated and above 100x. If speed is among your priorities, you should consider MLtraq to run your experiments. The primary disadvantage of MLtraq is its lack of a complete web dashboard.
# 

# In[ ]:




