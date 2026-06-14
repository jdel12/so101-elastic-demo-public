# Operate

This is where the value lives. The build and deploy stages get you a
running robot with observability. This stage is about using it — running
trials, understanding results, training better models, and diagnosing
failures systematically.

| Guide | What you'll learn |
|-------|-------------------|
| [01-trials.md](01-trials.md) | Running trials, interpreting results, common failure modes |
| [02-training.md](02-training.md) | Recording episodes, training models, evaluation, hot-swap |
| [03-diagnostics.md](03-diagnostics.md) | Model Decision Observatory, failure classification, forward kinematics |
| [04-gpu-profiling.md](04-gpu-profiling.md) | Kernel-level GPU analysis, component decomposition, cross-model comparison |

The diagnostic and profiling tools are what make this project different from
a standard robotics setup. They answer *why* the model fails and *what* to
change — not just *whether* it worked.
