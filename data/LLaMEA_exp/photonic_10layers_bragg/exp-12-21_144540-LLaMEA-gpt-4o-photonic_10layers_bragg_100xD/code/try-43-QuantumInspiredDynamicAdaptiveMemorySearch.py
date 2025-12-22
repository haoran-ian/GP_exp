import numpy as np

class QuantumInspiredDynamicAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)
        self.memory = []
        self.iteration = 0
        self.adaptivity_rate = 0.1
        self.diversity_threshold = 0.2 * (np.max(self.dim) - np.min(self.dim)) 

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
            memory_solution, memory_value = self._select_from_memory()

            # Update adaptivity rate based on difference from best
            self.adaptivity_rate = min(0.5, self.adaptivity_rate + 0.01 * (best_value - memory_value))

            # Quantum-inspired exploration with adaptive quantum fluctuations
            candidate_solution = self._quantum_explore(lb, ub)
            candidate_value = func(candidate_solution)

            self.iteration += 1

            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Update the memory with diversity preservation
            self._update_memory(candidate_solution, candidate_value)

        return best_solution

    def _select_from_memory(self):
        probabilities = np.array(
            [self.adaptivity_rate if i == 0 else np.random.rand() for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub):
        superposition = np.mean([sol for sol, _ in self.memory], axis=0)
        fluctuation = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb)
        candidate_solution = superposition + fluctuation
        return np.clip(candidate_solution, lb, ub)

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])
        if len(self.memory) > self.memory_size:
            self.memory.pop()
        
        # Ensure diversity
        if self._calculate_diversity(solution) < self.diversity_threshold:
            self.memory.pop()

    def _calculate_diversity(self, new_solution):
        distances = [np.linalg.norm(new_solution - sol) for sol, _ in self.memory]
        return np.mean(distances)