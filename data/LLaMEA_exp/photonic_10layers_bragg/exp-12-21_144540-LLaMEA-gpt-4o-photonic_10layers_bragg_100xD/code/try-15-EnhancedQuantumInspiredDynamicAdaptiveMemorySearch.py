import numpy as np

class EnhancedQuantumInspiredDynamicAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)  # Fixed memory size
        self.memory = []  # Store solutions
        self.iteration = 0
        self.adaptivity_rate = 0.1  # Initial adaptivity rate for exploration/exploitation balance

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
            # Dynamic prioritization in memory selection
            memory_solution, memory_value = self._select_from_memory()

            # Adaptive exploration strategy
            self.adaptivity_rate = max(0.01, min(0.5, self.adaptivity_rate + 0.01 * (best_value - memory_value)))

            # Quantum-inspired exploration with adaptive variance
            candidate_solution = self._quantum_explore(lb, ub, memory_solution)
            candidate_value = func(candidate_solution)

            self.iteration += 1

            # Update the best found solution
            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Update the memory
            self._update_memory(candidate_solution, candidate_value)

        return best_solution

    def _select_from_memory(self):
        # Dynamic prioritization: prefer better solutions adaptively
        probabilities = np.array([1.0 if i == 0 else self.adaptivity_rate for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub, base_solution):
        # Generate candidate with adaptive variance around a base solution
        superposition = np.mean([sol for sol, _ in self.memory], axis=0)
        entanglement = np.random.normal(0, self.adaptivity_rate, self.dim) * (ub - lb)
        candidate_solution = base_solution + entanglement
        return np.clip(candidate_solution, lb, ub)

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])  # Sort by value (objective)
        if len(self.memory) > self.memory_size:
            self.memory.pop()