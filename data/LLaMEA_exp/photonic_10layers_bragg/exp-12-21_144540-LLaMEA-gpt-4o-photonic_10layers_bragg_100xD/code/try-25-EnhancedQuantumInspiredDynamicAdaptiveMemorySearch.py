import numpy as np

class EnhancedQuantumInspiredDynamicAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.primary_memory_size = min(10, dim)
        self.secondary_memory_size = min(20, 2 * dim)
        self.primary_memory = []  # High-quality solutions
        self.secondary_memory = []  # Diverse solutions
        self.iteration = 0
        self.adaptivity_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')

        # Initialize with random solutions
        for _ in range(self.primary_memory_size):
            solution = np.random.uniform(lb, ub, self.dim)
            value = func(solution)
            self._update_memory(solution, value)
            if value < best_value:
                best_value = value
                best_solution = solution

        self.iteration += self.primary_memory_size

        while self.iteration < self.budget:
            # Select a solution from primary memory for exploitation
            memory_solution, memory_value = self._select_from_primary_memory()

            # Update adaptivity rate based on convergence
            self.adaptivity_rate = min(0.5, self.adaptivity_rate + 0.01 * (best_value - memory_value))

            # Quantum-inspired exploration with dynamic entanglement
            candidate_solution = self._quantum_explore(lb, ub)
            candidate_value = func(candidate_solution)

            self.iteration += 1

            # Update the best found solution
            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Update the memory
            self._update_memory(candidate_solution, candidate_value)

        return best_solution

    def _select_from_primary_memory(self):
        probabilities = np.array([1 / (i + 1) for i in range(len(self.primary_memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.primary_memory), p=probabilities)
        return self.primary_memory[idx]

    def _quantum_explore(self, lb, ub):
        if len(self.secondary_memory) > 0:
            superposition = np.mean([sol for sol, _ in self.secondary_memory], axis=0)
        else:
            superposition = np.mean([sol for sol, _ in self.primary_memory], axis=0)
        
        # Dynamic entanglement adjusted to convergence stage
        entanglement_strength = (1 - self.iteration / self.budget) * self.adaptivity_rate
        entanglement = np.random.uniform(-entanglement_strength, entanglement_strength, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub).flatten()

    def _update_memory(self, solution, value):
        self.secondary_memory.append((solution, value))
        self.secondary_memory.sort(key=lambda x: x[1])
        if len(self.secondary_memory) > self.secondary_memory_size:
            self.secondary_memory.pop()

        if len(self.primary_memory) < self.primary_memory_size or value < self.primary_memory[-1][1]:
            self.primary_memory.append((solution, value))
            self.primary_memory.sort(key=lambda x: x[1])
            if len(self.primary_memory) > self.primary_memory_size:
                self.primary_memory.pop()