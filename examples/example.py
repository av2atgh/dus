from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from dus import DependencyUpriseStructure, fishplot


DATA_DIR = Path(__file__).parent / "data"


def main():
    n_projects = 2
    task = [pd.read_parquet(DATA_DIR / f"task_{i + 1}.parquet") for i in range(n_projects)]
    task_rel = [pd.read_parquet(DATA_DIR / f"task_rel_{i + 1}.parquet") for i in range(n_projects)]
    dus = [
        DependencyUpriseStructure(task_rel[i].pred_task_id, task_rel[i].succ_task_id)
        for i in range(n_projects)
    ]

    fig, ax = plt.subplots(2, 2)

    label_map = {
        '00': '(A) Project 1',
        '10': '(C) Project 1',
        '01': '(B) Project 2',
        '11': '(D) Project 2',
    }

    i = 0
    for j in range(n_projects):
        ax[i, j].set_title(label_map[f'{i}{j}'], loc='left', x=0, y=1.05)
        fishplot(ax[i, j], dus[j].uprise_edges)

    i = 1
    for j in range(n_projects):
        ax[i, j].set_title(label_map[f'{i}{j}'], loc='left', x=0, y=1.05)
        dus_list, activities_per_dus = np.unique(dus[j].uprise, return_counts=True)
        task_property = 'physical_complete_pct'
        df = dus[j].get_nodes_data_aggregated_by_uprise(
            task[j][['task_id', task_property]],
            'task_id',
            {task_property: 'mean'},
        )
        fishplot(
            ax[i, j],
            dus[j].uprise_edges,
            x=dus_list,
            y=activities_per_dus,
            z=df[task_property].round(2).values,
        )

    fig.colorbar(
        ax[1, 1].scatter([0, 0], [0, 1], c=[0, 100], cmap='turbo', s=0),
        ax=ax,
        orientation='horizontal',
        fraction=.4,
        label='Progress %',
    )

    plt.subplots_adjust(bottom=0.5, right=1, wspace=0.1, top=1.5, hspace=0.1)

    out = Path(__file__).parent / "example_output.png"
    fig.savefig(out, bbox_inches='tight', dpi=120)
    print(f"Saved figure to {out}")


if __name__ == "__main__":
    main()
