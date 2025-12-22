import numpy as np

class QuantumInspiredHierarchicalDynamicMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(15, dim)  # Increased memory size for diversity
        self.memory = []  # Store solutions
        self.iteration = 0
        self.adaptivity_rate = 0.1  # Initial adaptivity rate
        self.hierarchy_levels = 3  # Hierarchical memory levels

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
            # Select a solution from hierarchical memory levels
            memory_solution, memory_value = self._select_from_hierarchical_memory()

            # Update adaptivity rate based on differences from best
            self.adaptivity_rate = min(0.5, self.adaptivity_rate + 0.01 * np.abs(best_value - memory_value))

            # Quantum-inspired exploration with self-adaptive learning rate
            candidate_solution = self._quantum_explore(lb, ub)
            candidate_value = func(candidate_solution)

            self.iteration += 1

            # Update the best found solution
            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Update the memory with dynamic filtering
            self._update_memory(candidate_solution, candidate_value)

        return best_solution

    def _select_from_hierarchical_memory(self):
        # Hierarchical memory selection for diverse exploitation
        level_probabilities = np.linspace(0.5, 0.1, self.hierarchy_levels) / np.sum(np.linspace(0.5, 0.1, self.hierarchy_levels))
        level = np.random.choice(self.hierarchy_levels, p=level_probabilities)
        level_size = len(self.memory) // self.hierarchy_levels
        level_start = level * level_size
        level_end = min((level + 1) * level_size, len(self.memory))
        
        # Prefer better solutions within the selected level
        probabilities = np.array([self.adaptivity_rate if i == 0 else np.random.rand() for i in range(level_start, level_end)])
        probabilities /= probabilities.sum()
        idx = np.random.choice(range(level_start, level_end), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub):
        # Quantum superposition to generate new candidate solution
        superposition = np.mean([sol for sol, _ in self.memory], axis=0)
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub)

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])  # Sort by value (objective)
        if len(self.memory) > self.memory_size:
            self.memory.pop()