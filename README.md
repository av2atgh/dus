# dus

**Dependency Uprise Structure (DUS)** — a representation of a directed acyclic
graph (DAG) that groups activities into discrete *uprises* (longest-path levels)
and exposes the inter-uprise dependency network. Useful for visualising and
analysing project plans, build graphs, scheduling networks, or any DAG where
levels and progression matter.

The package also ships `fishplot`, a polar-coordinate visualisation of the
uprise structure.

## Repository layout

```
.
├── dus/                     # Python package
│   ├── __init__.py
│   ├── dus.py               # DependencyUpriseStructure class
│   └── fishplot.py          # fishplot visualisation
├── examples/
│   ├── example.py           # runnable end-to-end example
│   └── data/                # sample parquet inputs
│       ├── task_1.parquet
│       ├── task_2.parquet
│       ├── task_rel_1.parquet
│       └── task_rel_2.parquet
├── analysis/                # analysis behind the paper (see below)
│   ├── stats.py             # model selection, bootstrap CIs, commonality tests
│   ├── psplib.py            # PSPLIB download, DUS/DPS, public replication
│   ├── recompute.py         # exact per-project statistics from raw schedules
│   └── make_figures.py      # every figure in the paper
├── manuscript/              # paper sources, figures and compiled PDFs
├── pyproject.toml
├── requirements.txt
├── LICENSE
└── README.md
```

## Installation

```bash
pip install -r requirements.txt
# or, install the package in editable mode:
pip install -e .
```

## Quick start

```python
from dus import DependencyUpriseStructure, fishplot
import matplotlib.pyplot as plt

dus = DependencyUpriseStructure(sources, targets)  # arrays of edge endpoints
fig, ax = plt.subplots()
fishplot(ax, dus.uprise_edges)
plt.show()
```

## Running the example

```bash
python examples/example.py
```

This loads two sample project DAGs from `examples/data/`, builds the DUS for
each, and renders a 2x2 fishplot figure (saved as
`examples/example_output.png`).

## License

See [LICENSE](LICENSE).

## Reproducing the paper

`analysis/` contains everything behind *High-Level Structure of Dependency
Networks*. Each script can be run on its own from the repository root.

```bash
python analysis/psplib.py        # public data only -- no extra inputs needed
python analysis/stats.py         # needs the proprietary summary CSV (see below)
python analysis/make_figures.py  # writes fig1.pdf ... fig6.pdf to manuscript/
```

| Script | What it does |
| --- | --- |
| `stats.py` | Fits four functional forms under additive and multiplicative errors, compares them by AIC on a common likelihood scale, computes bootstrap confidence intervals, runs Breusch--Pagan diagnostics, and tests parameter commonality across companies with nested likelihood ratio tests. Writes `results.json` and `table1.tex`. |
| `psplib.py` | Downloads the four single-mode PSPLIB sets (2040 instances), extracts each precedence DAG, computes the DUS and DPS decompositions, and refits the scaling relations. Also analyses the two anonymised schedules in `examples/data/`. Writes `psplib_instances.csv` and `psplib_results.json`. |
| `recompute.py` | Rebuilds the per-project summary table directly from raw schedules with exact integer DPS labels. Written after a float-precision defect was found in the original pipeline (see below). |
| `make_figures.py` | Regenerates all six figures from the two result files. |

### A note on DPS labels and precision

The DPS label packs one bit per DUS level. It must be built with exact integer
arithmetic. An earlier version of `dependency_fibration_structure` accumulated
it with `math.pow`, i.e. in float64, whose mantissa holds only 53 bits; beyond
53 uprise levels distinct predecessor patterns silently collapsed onto the same
value and the count of DPS classes was under-estimated, by up to 1.51x on real
schedules. This is fixed, and `recompute.py` regenerates the affected summary
statistics from the raw data.

### Data availability

`psplib.py` is fully self-contained: it downloads the benchmark instances on
first run (cached in `analysis/psplib_cache/`, which is not tracked) and also
uses the two anonymised real schedules in `examples/data/`.

`stats.py` and the parts of `make_figures.py` that plot the construction data
need a per-project summary CSV derived from proprietary Nodes & Links
schedules. That file is **not** distributed here. Point the script at it with:

```bash
export DUS_SUMMARY_CSV=/path/to/schedule_summary.csv
```

The committed `analysis/results.json` records the output of that run, so the
figures and every number quoted in the paper can be inspected without access to
the underlying schedules. The three contributing companies are anonymised as
blue, green and orange; the colour assignment is derived from the data (by
ascending project count), so no client identifiers appear anywhere in this
repository.
