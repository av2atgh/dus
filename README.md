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
