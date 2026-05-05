import numpy as np
import pandas as pd
from math import pow

class DependencyUpriseStructure:
    """A class for the Dependency Uprise Structure (DUS) of a directed acyclic graph"""
    
    def __init__(self, sources, targets):
        # map sources/targets to consecutive range or natural numbers
        source_targets = np.concatenate([sources, targets])
        unique_input_id, inverse_indices = np.unique(source_targets, return_inverse=True)
        n_sources = len(sources)
        sources_ = inverse_indices[:n_sources]
        targets_ = inverse_indices[n_sources:]
        n_nodes = len(unique_input_id)
        nodes = range(n_nodes)
        
        # get topological sort
        sort_list, successors = self.topological_sort(sources_, targets_)

        # activity dus
        uprise = np.zeros(n_nodes, dtype=np.int32)
        for i in sort_list:
            if i in successors:
                uprise[successors[i]] = np.maximum(uprise[successors[i]], uprise[i] + 1)
        n_uprises = uprise.max() + 1

        # predecessor/successor uprise set
        activity_predecessors_uprise_set = [set() for i in nodes]
        activity_successors_uprise_set = [set() for i in nodes]
        uprise_edges = {}
        uprise_predecessors_set = [set() for i in range(n_uprises)]
        uprise_successors_set = [set() for i in range(n_uprises)]
        for i in sort_list:
            if i in successors:
                ui = uprise[i]
                for j in successors[i]:
                    uj = uprise[j]
                    uprise_predecessors_set[uj].add(ui)
                    uprise_successors_set[ui].add(uj)
                    activity_predecessors_uprise_set[j].add(ui)
                    activity_successors_uprise_set[i].add(uj)
                    if (ui, uj) in uprise_edges:
                        uprise_edges[(ui, uj)] += 1
                    else:
                        uprise_edges[(ui, uj)] = 1
           
        # self variables
        self.n_nodes = n_nodes
        self.n_uprises = n_uprises
        self.successors = successors
        self.sort_list = sort_list
        self.unique_input_id = unique_input_id
        self.uprise = uprise
        self.activity_predecessors_uprise_set = activity_predecessors_uprise_set
        self.activity_successors_uprise_set = activity_successors_uprise_set
        self.uprise_predecessors_set = uprise_predecessors_set
        self.uprise_successors_set = uprise_successors_set
        self.uprise_edges = uprise_edges

    def dependency_fibration_structure(self):
        fibration = []
        for i in range(self.n_nodes):
            fibration_ = 0
            for j in self.activity_predecessors_uprise_set[i]:
                fibration_ += pow(2, j)
            fibration.append(fibration_)
        self.fibration = fibration
        self.n_fibrations = len(set(fibration))
                
    def get_uprise_average_distance_to_end(self) -> float:
        uprise_distance_to_end = np.zeros(self.n_uprises, dtype=np.int32)
        for i in range(self.n_uprises-1)[::-1]:
            uprise_distance_to_end[i] = np.min(uprise_distance_to_end[list(self.uprise_successors_set[i])]) + 1
        return np.mean(uprise_distance_to_end)

    def get_nodes_data_aggregated_by_uprise(self, nodes_data: pd.DataFrame, node_id, aggregate_method: dict):
        for c in nodes_data.columns:
            if 'date' in c:
                nodes_data[c] = pd.to_datetime(nodes_data[c], errors='coerce')
        node_table = pd.DataFrame({node_id: self.unique_input_id, 'dus': self.uprise})
        merge_columns = [node_id] + list(aggregate_method.keys())
        node_table = node_table.merge(nodes_data[merge_columns], how='left', on=node_id)
        return node_table.groupby('dus').agg(aggregate_method).reset_index() 

    @staticmethod
    def topological_sort(sources: np.ndarray, targets: np.ndarray) -> list:
        """Finds the topological sort of a directed graph using Kahn's algorithm"""

        # get out-neighbours
        argsort = sources.argsort()
        sources = sources[argsort]
        targets = targets[argsort]
        unique_sources, indices = np.unique(sources, return_index=True)
        out_neighbours = dict(zip(unique_sources, np.split(targets, indices[1:])))

        # get in-degrees
        unique_targets, counts = np.unique(targets, return_counts=True)
        in_degree = dict(zip(unique_targets, counts))
        n_targets = len(unique_targets)

        # get topological sort
        seed_list = list(set(unique_sources) - set(unique_targets))
        sort_list = []
        while seed_list:
            s = seed_list.pop(0)
            sort_list.append(s)
            if s in out_neighbours:
                for t in out_neighbours[s]:
                    in_degree[t] -= 1
                    if in_degree[t] == 0:
                        seed_list.append(t)
                        n_targets -= 1

        sort_list = list() if n_targets > 0 else sort_list

        return sort_list, out_neighbours
