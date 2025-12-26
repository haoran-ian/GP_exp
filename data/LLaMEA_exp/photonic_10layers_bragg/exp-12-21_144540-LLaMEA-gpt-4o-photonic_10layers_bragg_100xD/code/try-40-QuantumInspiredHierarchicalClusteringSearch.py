import numpy as np

class QuantumInspiredHierarchicalClusteringSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)  # Fixed memory size
        self.memory = []  # Store solutions
        self.iteration = 0
        self.adaptivity_rate = 0.1  # Initial adaptivity rate for exploration/exploitation balance
        self.cluster_threshold = 0.2  # Threshold for clustering similar solutions

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')

        # Initialize with random solutions
        for _ in range(self.memory_size):
            solution = np.random.uniform(lb, ub, self.dim)
            value = func(solution)
            self._update_memory(solution, value)
            if value < best_value:
                best_value = value
                best_solution = solution

        self.iteration += self.memory_size

        while self.iteration < self.budget:
            # Cluster memory to select diverse solutions
            clustered_memory = self._cluster_memory()
            memory_solution, memory_value = self._select_from_cluster(clustered_memory)

            # Update adaptivity rate based on difference from best
            self.adaptivity_rate = min(0.5, self.adaptivity_rate + 0.01 * (best_value - memory_value))

            # Quantum-inspired exploration with adaptive phase adjustment
            candidate_solution = self._quantum_explore_with_phase(lb, ub)
            candidate_value = func(candidate_solution)

            self.iteration += 1

            # Update the best found solution
            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Update the memory with dynamic filtering
            self._update_memory(candidate_solution, candidate_value)

        return best_solution

    def _cluster_memory(self):
        # Hierarchically cluster solutions in memory based on similarity
        clusters = []
        for sol, val in self.memory:
            added_to_cluster = False
            for cluster in clusters:
                if np.linalg.norm(sol - cluster[0][0]) < self.cluster_threshold:
                    cluster.append((sol, val))
                    added_to_cluster = True
                    break
            if not added_to_cluster:
                clusters.append([(sol, val)])
        return clusters

    def _select_from_cluster(self, clustered_memory):
        # Select the best solution from a randomly chosen cluster
        cluster = np.random.choice(clustered_memory)
        best_in_cluster = min(cluster, key=lambda x: x[1])
        return best_in_cluster

    def _quantum_explore_with_phase(self, lb, ub):
        # Quantum superposition with adaptive phase exploration
        superposition = np.array([np.mean([sol for sol, _ in self.memory], axis=0)])
        phase_shift = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim)
        candidate_solution = superposition + phase_shift * (ub - lb)
        return np.clip(candidate_solution, lb, ub).flatten()

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])  # Sort by value (objective)
        if len(self.memory) > self.memory_size:
            self.memory.pop()