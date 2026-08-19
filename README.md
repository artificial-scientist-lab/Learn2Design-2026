
# NeurIPS 2026 Challenge: Learn2Design-2026
[![License](https://img.shields.io/github/license/artificial-scientist-lab/Learn2Design-2026)](https://github.com/artificial-scientist-lab/Learn2Design-2026/blob/main/LICENSE)
[![image](https://img.shields.io/pypi/pyversions/dfbench.svg)](https://pypi.python.org/pypi/dfbench)
[![GitHub](https://img.shields.io/badge/GitHub-dfbench-3232c8?logo=github&logoColor=white)](https://github.com/artificial-scientist-lab/Differometor-Benchmark)
[![Static Badge](https://img.shields.io/badge/Codabench-Submit-4CB2AD)](https://submit.learn2design2026.com/competitions/4/)


## A physics experiment design competition for gravitational-wave detectors

<p align="center">
Jonathan Klimesch<sup>1,2</sup>, Laurin Sefa<sup>1,3</sup>, Soham Basu<sup>1</sup>, Priya Kanagasabapathi<sup>1</sup>,<br>
Sören Arlt<sup>1,2</sup>, Xuemei Gu<sup>4</sup>, Thomas Christie<sup>1</sup>, Colin Doumont<sup>1</sup>,<br>
Andreas Freise<sup>5</sup>, Rana Adhikari<sup>6</sup>, Philipp Hennig<sup>1</sup>, Mario Krenn<sup>1</sup>
</p>

<p align="center">
<sup>1</sup>Department for Computer Science, Faculty of Science, University of Tübingen, Tübingen, Germany<br>
<sup>2</sup>Feyer, Tübingen, Germany<br>
<sup>3</sup>Zuse School ELIZA, Darmstadt, Germany<br>
<sup>4</sup>Institut für Festkörpertheorie und Optik, Friedrich-Schiller-Universität Jena, Jena, Germany<br>
<sup>5</sup>Nikhef, National Institute for Subatomic Physics, Amsterdam, The Netherlands<br>
<sup>6</sup>Institute for Quantum Information and Matter, California Institute of Technology, Pasadena, CA, USA
</p>

<p align="center">
  <img src="learn2design2026_with_neurips.jpg" alt="Learn2Design-2026 with NeurIPS" width="720">
</p>

**Learn2Design-2026** is a NeurIPS 2026 challenge on the automated design of highly sensitive [gravitational-wave detectors](https://en.wikipedia.org/wiki/LIGO) under realistic experimental constraints.

Participants are given [a search space of gravitational-wave detectors](#quasi-universal-interferometer-uifo).
Within this search space, the task is to optimize roughly **200 continuous parameters**, such as laser powers, mirror reflectivities, grid distances, and related experimental degrees of freedom.

The goal is to develop an algorithm that **maximizes detector sensitivity** while satisfying physical and experimental constraints, all within a fixed compute budget. Participants submit the algorithm itself, not only a final design. The submitted algorithms will be run by the organizers on standardized local hardware and ranked by their average performance on hidden detector topologies.

The challenge provides the differentiable, JAX-based simulator **[Differometor](https://github.com/artificial-scientist-lab/Differometor)**. Its objective function is pure, JAX-compatible, and supports gradients and Hessians through automatic differentiation, enabling gradient-based, hybrid, and learning-based optimization strategies.

To support algorithm development, we also provide approximately **30,000 high-quality detector designs** generated through a **360,000 GPU-hour EuroHPC exploration campaign**. These examples can be used for supervised learning, initialization, representation learning, generative modeling, benchmarking, or other exploration and optimization approaches.

Algorithms will be ranked by their hidden-evaluation performance, with **EUR 25,000** in prize money (sponsored by [SPRIND](https://www.sprind.org/)).

Beyond gravitational-wave detection, Learn2Design-2026 asks a broader scientific question:
**Can AI systems discover scientific instruments that go beyond human intuition while remaining physically meaningful and experimentally constrained?**

More Infos in the accepted proposal for [Learn2Design-2026](Learn2Design_details.pdf).

## Prize money

- **1st prize: EUR 10,000**
- **2nd prize: EUR 6,000**
- **3rd prize: EUR 3,000**
- **Two special prizes:  EUR 3,000** (judged by a committee for the most surprising or creative
solution, and simplest strong-performing solution).

Prize eligibility requires the submission of an, initally confidential, *short technical report of 2-4 pages* (see below).

The prize money is sponsored by [SPRIND (Federal Agency for Disruptive Innovation / Bundesagentur für Sprunginnovationen)](https://www.sprind.org/).

### Eligibility

Participation is open worldwide, subject to applicable laws and the eligibility requirements of NeurIPS and SPRIND. Participants based in Russia are not eligible to receive prize money. Individuals and institutions subject to applicable sanctions, including those on the U.S. SDN list, are also ineligible.

## Technical reports and post-competition publication

- After the final hidden evaluation, we will invite all teams whose final submissions
outperform the organizer-provided baseline threshold to **submit a short technical
report of 2-4 pages** describing their method. Timely submission of this report is
required for organizational reasons and is a prerequisite for prize eligibility,
special-prize consideration, workshop-presentation selection, and participation
in the joint post-competition publication.

- The **technical reports** help the organizers verify and understand the submitted methods,
prepare the workshop program, document the scientific and algorithmic lessons of
the competition, report to the sponsor, and prepare a joint post-competition
analysis.

- Technical reports will **initially be submitted confidentially** to the organizers.
We may request these reports before teams publicly release their own method
descriptions, so that the organizers can coordinate the competition analysis and
the joint publication. Reports will not be made public by the organizers without
author approval.

- Participants **retain copyright** in their own reports and methods.

- Teams with eligible final submissions will be **invited to contribute to a joint
competition-review paper** as named authors. The short technical report will serve
as the starting point for describing each team's method in this joint analysis.


## How it works

- You submit a ZIP archive containing your optimization algorithm, dependency
  requirements, and any supporting files through the [submission portal](https://submit.learn2design2026.com/).
- You can submit as often as you want; for each monthly evaluation we use the last submission before that month's deadline.
- Each evaluation runs it on 10 new held-out hidden topologies on a single A100 GPU with an AMD EPYC 7302 CPU.
- Time budget per topology: exactly 4 hours of wall-clock time after `objective.start_logging()`.
- The best loss among setups [satisfying all constraints](docs/dfbench_overview.md#power-constraints-and-aux-diagnostics) will be recorded for each of the 10 runs.
- If a run has no constraint-satisfying setup, we use the best constraint-satisfying loss from random search on that topology instead.
- The arithmetic mean over the 10 recorded losses is your score. Lower is better.
- Your public score is published to the monthly leaderboard; the final evaluation uses its own private hidden topologies.

A "[topology](docs/dfbench_overview.md#explanation-of-topology)" fixes the choice of optical components for an experimental ansatz; you only optimize
the continuous parameters attached to it. These could be laser power, mirror 
reflectivity, grid distance, etc.

Your algorithm is allowed to evaluate the objective in batches via `jax.vmap` (the `obj.vmap_*` methods). The whole batch runs in a single vmapped forward pass, which saves significant time per element opposed to looping single evals. This is encouraged for population-based methods (PSO, CMA-ES, evolutionary strategies) and any algorithm that naturally evaluates multiple candidates per step.


## Getting Started

A submission's `submission.py` file defines exactly one class subclassing
`OptimizationAlgorithm`:

```python
from dfbench import Objective, OptimizationAlgorithm
from jaxtyping import Array, Float


class MyAlgorithm(OptimizationAlgorithm):

    algorithm_str = "my_algo"

    def optimize(
        self,
        objective: Objective,
        init_params: Float[Array, "..."] | None = None,
        random_seed: int | None = None,
        **kwargs,
    ) -> None:
        # 1. Configure the search space and apply random seeds
        self.prepare(objective, unbounded=False, random_seed=random_seed)

        params = init_params if init_params is not None else objective.random_params()

        # 2. Warm up JIT (compilation is not counted against the budget)
        objective.warmup_value()  # Use any other warmup_* method if desired

        # 3. Start the clock
        objective.start_logging()

        # 4. Optimization loop
        while not objective.budget_exceeded:
            # ... YOUR update logic here, producing `params` ...
            loss = objective.value(params)  # automatically logged
```

With `dfbench 0.3.3`, result-producing evaluation methods, raw callable getters,
and `log_evaluation()` require `start_logging()` first. Documented problem
context such as bounds, dimension, `problem_spec`, and `optimization_pairs`, as
well as random sampling, pre-run setters, and Objective-provided `warmup_*()`
methods, remain available before logging.

That is the entire algorithm contract. The `Objective` handles seeding, history,
checkpointing, and budget enforcement. You write the loop, place the class in
`submission.py`, and package it as described in [Submitting](#submitting).
Execution examples of such classes are provided in
[`learn2design/scripts/`](learn2design/scripts/) and below.

### 1. Test with Constrained Voyager (Fast)
<details>
<summary> How to execute </summary>

<br>
If you are new to the challenge, we recommend starting with the lighter `ConstrainedVoyagerProblem` (look at the script [`cvoyager_adam_gd.py`](learn2design/scripts/cvoyager_adam_gd.py) for an execution example). It uses the exact same `Objective` API and loss calculation but is a smaller, faster problem, making it great for quickly testing your optimization loop or getting a feel for the performance:

```python
from dfbench.problems import ConstrainedVoyagerProblem
from dfbench import Objective
from learn2design.example_algorithms import MyAlgorithm  # Replace with your own algorithm or one of the examples

problem = ConstrainedVoyagerProblem()
objective = Objective(problem, max_time=2*60)  # 2 Minutes of optimization

optimizer = MyAlgorithm()
optimizer.optimize(objective)
```
</details>

### 2. Scale up to UIFO (The Competition Target)
<details>
<summary> How to execute </summary>
<br>
The `UIFOProblem` is the actual target of this competition. Once your algorithm runs successfully on the smaller problem, scale it up to the Quasi-Universal Interferometer (UIFO).

To see an example of execution on the full problem, look at scripts like [`uifo_random_search.py`](learn2design/scripts/uifo_random_search.py) or [`uifo_adam_gd.py`](learn2design/scripts/uifo_adam_gd.py). The execution looks like this:

```python
from dfbench.problems import UIFOProblem
from dfbench import Objective
from learn2design.example_algorithms import MyAlgorithm  # Replace with your own algorithm or one of the examples

problem = UIFOProblem(topology_seed=42)  # Random topology with seed 42
objective = Objective(problem, max_time=10*60)  # 10 Minutes of optimization

optimizer = MyAlgorithm()
optimizer.optimize(objective)
```
</details>

<br>

You can find most functionality of the Objective API at [`docs/dfbench_overview.md`](docs/dfbench_overview.md). For a simple explanation of the UIFO loss, see [What objective function is optimized for UIFO?](docs/dfbench/FAQ.md#what-objective-function-is-optimized-for-uifo) or open the [interactive sensitivity loss explorer](docs/sensitivity_loss_explorer.html). In particular, be aware that constrained problems let you [change the penalty function](docs/dfbench_overview.md#power-constraints-and-aux-diagnostics) used during optimization.

## Install

Requires Python 3.11 or newer.

```bash
git clone https://github.com/artificial-scientist-lab/Learn2Design-2026
cd Learn2Design-2026
pip install -e .
```

If you want GPU support, make sure you have CUDA 12 or 13 installed:
```bash
pip install -e ".[cuda13]" # or ".[cuda12]"
```

Via uv:
```bash
uv sync --extra cuda13
```

This pulls in [`dfbench`](https://github.com/artificial-scientist-lab/Differometor-Benchmark) (v0.3.3, the benchmark
framework). `dfbench` in turn uses
[`differometor`](https://github.com/artificial-scientist-lab/Differometor), the JAX-based
interferometer simulator.

Smoke-test one UIFO evaluation (may take a few minutes to JIT-compile) with [smoke_test.py](learn2design/scripts/smoke_test.py):
```bash
python learn2design/scripts/smoke_test.py
```


## Quasi-Universal Interferometer (UIFO)

The given search space of gravitational-wave detectors is visualized below. It consists of a grid structure which can hold different combinations of five building blocks. The beam splitter and directional beam splitter blocks can fill the grid centers (in any 90° rotation). The laser, squeezer, and detector blocks can fill the boundary cells, whereas the detector block can only be placed once.

Each component has parameters that can be optimized within certain
[ranges](docs/dfbench_overview.md#bounds).

For the topology string format, component-code mapping, and ways to instantiate
UIFO topologies directly, see [Explanation of "Topology"](docs/dfbench_overview.md#explanation-of-topology).

The goal is to find algorithms that work well on any UIFO topology sampled from this search space; two example topologies are visualized in the figure below. Each evaluation runs on its own 10 hidden topologies.

<p align="center">
  <img src="media/UIFO.png" alt="Quasi-Universal Interferometer (UIFO)" width="720">
</p>


## Dataset

The precomputed UIFO design corpus is available in [`dataset/`](dataset/).
It contains [`dataset.h5`](dataset/dataset.h5),
a compact HDF5 dataset with 29,650 pure-broadband optimized setups. Each entry
stores a topology string, bounded parameter vector, saved loss, sensitivity
curve, power data, complexity, and metadata such as `unique_hash`. The folder
also includes a standalone Plotly HTML swarmplot for browsing losses and topology
groups interactively.

Start with the dataset-specific guide in [`dataset/README.md`](dataset/README.md).
It documents the HDF5 layout, efficient lazy slicing of parameter and power
pools, and includes runnable examples for loading, evaluating, and visualizing
an entry with `UIFOProblem` and Differometor.

```bash
python dataset/examples/load_entry.py --index 0
python dataset/examples/evaluate_entry.py --index 0
python dataset/examples/visualize_entry.py --index 0
```

The dataset was distilled and curated from the much larger [GraviTune Dataset](https://github.com/artificial-scientist-lab/GraviTune-Dataset).


## Repository layout

The repository is organized around a small number of entry points:

| Path | Purpose |
|---|---|
| `learn2design/` | Package code, including example algorithms and runnable scripts. |
| `dataset/` | Precomputed UIFO design corpus, dataset README, and loading/evaluation examples. |
| `docs/` | Competition docs plus the bundled `dfbench` reference pages. |
| `pyproject.toml` | Package metadata and dependency definitions. |

<details>
<summary>Show a more detailed layout</summary>

```text
learn2design/
├── example_algorithms/            # Reference implementations
│   ├── __init__.py
│   ├── adam_gd.py                 # Standard Adam optimizer
│   ├── na_adam_gd.py              # Adam with decaying Gaussian noise
│   ├── optax_sgdm.py              # SGD with momentum (Optax)
│   ├── scipy_bfgs.py              # BFGS (SciPy)
│   ├── lbfgs_gd.py                # L-BFGS (Optax)
│   ├── random_search.py           # Uniform random search
│   └── pycma_cmaes.py             # CMA-ES (pycma)
└── scripts/                       # Minimal runnable entry points
    ├── smoke_test.py              # Single-eval smoke test
    ├── uifo_adam_gd.py            # UIFO + AdamGD
    ├── uifo_na_adam_gd.py         # UIFO + NAAdamGD
    ├── uifo_optax_sgdm.py         # UIFO + OptaxSGDM
    ├── uifo_scipy_bfgs.py         # UIFO + BFGS
    ├── uifo_lbfgs_gd.py           # UIFO + LBFGSGD
    ├── uifo_random_search.py     # UIFO + RandomSearch
    ├── uifo_pycma_cmaes.py       # UIFO + PyCMACMAES
    └── cvoyager_adam_gd.py       # ConstrainedVoyager + AdamGD (lightweight)

docs/
├── dfbench_overview.md  # Overview of the functionality you need
├── submission.md        # Submission rules
├── scoring.md           # Scoring and leaderboard details
├── FAQ.md               # Competition FAQ
└── dfbench/             # dfbench 0.3.3 reference pages
    ├── Architecture-Overview.md
    ├── Objective-API-Reference.md
    ├── Problems.md
    ├── Algorithms.md
    ├── Implementing-a-New-Algorithm.md   # step-by-step guide for new algorithms
    ├── Utilities-and-Helpers.md
    └── FAQ.md

dataset/
├── dataset.h5
├── dataset_dashboard.html
├── README.md
└── examples/            # Loading, evaluation, and visualization examples
    ├── dataset_utils.py
    ├── load_entry.py
    ├── evaluate_entry.py
    └── visualize_entry.py
```

</details>


## Baselines

In the plots below, we provide comparisons between baselines from different classes of algorithms.

![Baseline category overview](media/category_algorithms_loss_mean_sem.png)

The table below summarizes the example baselines included in [`learn2design/example_algorithms`](learn2design/example_algorithms).

> [!TIP]
> For a host of other baselines, take a look at [dfbench/algorithms](https://github.com/artificial-scientist-lab/Differometor-Benchmark/tree/main/src/dfbench/algorithms)

Rows are ordered by displayed mean loss; ties in the rounded values are broken
by displayed SEM and then alphabetically.

Because this repository depends on `dfbench` as an external package, it does
not contain the `dfbench` source tree itself. The links below therefore open
the matching documented algorithm section in [`docs/dfbench/Algorithms.md`](docs/dfbench/Algorithms.md), using the exact class names and variants from that documentation.

A loss of zero means that the optimizer has discovered the best known human designed gravitational wave detector (within the same technical resources, such as arm lengths). **Losses below zero are possible and [expected](https://journals.aps.org/prx/abstract/10.1103/PhysRevX.15.021012)**.


| Rank & name | General type* | Detailed implementation | Average loss ± SEM | Link to example |
|---|---|---|---|---|
| 1. `AdamGD` | Gradient-based | Standard Adam optimizer utilizing gradient clipping for stability | 1.1 ± 0.3 | [AdamGD](learn2design/example_algorithms/adam_gd.py) |
| 2. `NAAdamGD` | Gradient-based | Adam optimizer enhanced with decaying Gaussian noise to escape local optima | 1.2 ± 0.4 | [NAAdamGD](learn2design/example_algorithms/na_adam_gd.py) |
| 3. `OptaxSGDM` | Gradient-based | Stochastic Gradient Descent (SGD) with momentum, implemented via Optax | 1.2 ± 0.4 | [OptaxSGDM](learn2design/example_algorithms/optax_sgdm.py) |
| 4. `BFGS` | Gradient-based | BFGS quasi-Newton method (SciPy) for gradient-based optimization | 1.8 ± 0.2 | [BFGS](learn2design/example_algorithms/scipy_bfgs.py) |
| 5. `LBFGSGD` | Gradient-based | Limited-memory BFGS (Optax) featuring a custom JIT-compiled logging loop | 2.9 ± 0.2 | [LBFGSGD](learn2design/example_algorithms/lbfgs_gd.py) |
| 6. `PyCMACMAES` | Evolutionary | Vanilla CMA-ES (pycma) searching in the unit cube, mapped to physical bounds at evaluation | 4.1 ± 0.1 | [PyCMACMAES](learn2design/example_algorithms/pycma_cmaes.py) |
| 7. `RandomSearch` | Global Search | Uniform random sampling baseline evaluated in batches within bounds | 4.8 ± 0.03 | [RandomSearch](learn2design/example_algorithms/random_search.py) |


*General types follow `dfbench`'s coarse `AlgorithmType` system:
gradient-based, evolutionary, surrogate-based, global_search, derivative_free and generative.


## Submitting

Upload a ZIP archive to the [submission portal](https://submit.learn2design2026.com/)
containing these two mandatory files at its root:

```text
submission.zip
├── submission.py       # Defines exactly one dfbench.OptimizationAlgorithm subclass
├── requirements.txt    # PEP 508 dependencies, one per line
└── ...                 # Optional modules, weights, or other supporting files
```

`submission.py` must define exactly one Python class that subclasses
`dfbench.OptimizationAlgorithm`. The evaluator calls:

```python
MyAlgorithm.optimize(...)
```

`requirements.txt` must list all required Python packages using
[PEP 508](https://peps.python.org/pep-0508/) syntax, with one dependency per
line:

```text
<package1>==1.2.3
<package2>==4.5.6
```

You may include Python modules, pretrained neural-network weights, or other
supporting files. The evaluation script runs from the root of the extracted
archive, so access bundled files using relative paths. See the
[submission rules](docs/submission.md) for the complete format, dependency
policy, time budget, and evaluation procedure.


## Timeline

| Date | Event |
|---|---|
| 09.07.2026 | Start of competition |
| 10.08.2026, 12:00 UTC | [Submission platform](https://submit.learn2design2026.com/) opens |
| 26.08.2026, AoE | First submission deadline |
| 12.09.2026, AoE | Second submission deadline |
| 29.09.2026, AoE | Third submission deadline |
| 15.10.2026, AoE | Final submission deadline |
| Before workshop | Private leaderboard announced |

The first three submission deadlines (**26 August 2026**, **12 September 2026**, and **29 September 2026**) are used only for the **public leaderboard**. These evaluations allow you to see how well your solution performs before the final submission deadline. Participation in these intermediate evaluations is **optional**. You do not need to submit a solution by any of these three deadlines to remain eligible for a prize.

The competition winners will be determined **only based on submissions received by the final deadline: 15 October 2026, Anywhere on Earth (AoE)**. In other words, even if you register on 15 October 2026 and make your first and only submission that same day, you are still fully eligible to win the competition and receive a prize.


## Contact

Whenever possible, we recommend communicating with the organizers via GitHub issues so that other participants with similar questions can also see the solutions. If your question cannot be discussed publicly, please contact [Jonathan](mailto:jonathan@feyer.ai), [Laurin](mailto:laurin.sefa@student.uni-tuebingen.de), [Priya](mailto:shanmugapriya.kanagasabapathi@uni-tuebingen.de), [Soham](mailto:soham.basu@uni-tuebingen.de), or [Mario](mailto:mario.krenn@uni-tuebingen.de).

## Resources

- **Website:** <https://www.learn2design2026.com/>
- **Submission portal:** <https://submit.learn2design2026.com/>
- **Repository:** <https://github.com/artificial-scientist-lab/Learn2Design-2026>
- **Issues / questions:** <https://github.com/artificial-scientist-lab/Learn2Design-2026/issues>
- **Dataset guide:** [`dataset/README.md`](dataset/README.md)
- **Simulator:** [`differometor`](https://pypi.org/project/differometor/)
- **Benchmark framework:** [`dfbench`](docs/dfbench/Architecture-Overview.md)
- **Group:** [Artificial Scientist Lab](https://www.artificial-scientist-lab.ai/)



> [!TIP]
> * [docs/dfbench_overview](docs/dfbench_overview.md) gives a brief overview of all the functionality provided by the Objective and the [dfbench](https://github.com/artificial-scientist-lab/Differometor-Benchmark) package in general.
> * Check out [Submission](docs/submission.md) and [Scoring](docs/scoring.md) for further details on the submission system and scoring criteria we use in this competition, respectively.
> * Take a look at the [FAQs](docs/FAQ.md) which might help answer any further questions regarding Learn2Design-2026.
> * [docs/dfbench](docs/dfbench/) includes a comprehensive documentation of the __dfbench__ package.  


## Citing

```bibtex
@misc{learn2design2026,
  title  = {Learn2Design 2026: A Physics Experiment Design Competition for Gravitational-Wave Detectors},
  author = {Klimesch, Jonathan and Sefa, Laurin and Basu, Soham and Kanagasabapathi, Priya and Arlt, S{\"o}ren and Gu, Xuemei and Christie, Thomas and Doumont, Colin and Freise, Andreas and Adhikari, Rana and Hennig, Philipp and Krenn, Mario},
  year   = {2026},
  url    = {https://github.com/artificial-scientist-lab/Learn2Design-2026}
}
```

## License

MIT — see [LICENSE](LICENSE).
