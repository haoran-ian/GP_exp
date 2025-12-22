import numpy as np

class EnhancedQuantumInspiredDynamicAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)
        self.memory = []
        self.iteration = 0
        self.adaptivity_rate = 0.1
        self.convergence_rate = 0.01
        self.successful_iterations = 0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')

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

            self.adaptivity_rate = min(0.5, self.adaptivity_rate + 0.01 * (best_value - memory_value))
            self.convergence_rate = 0.5 * (np.log1p(self.iteration) / self.budget)

            candidate_solution = self._quantum_explore(lb, ub)
            candidate_value = func(candidate_solution)
            self.iteration += 1

            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution
                self.successful_iterations += 1

            self._update_memory(candidate_solution, candidate_value)
            self._dynamically_resize_memory()

        return best_solution

    def _select_from_memory(self):
        probabilities = np.array([1.0 if i == 0 else self.adaptivity_rate for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub):
        superposition = np.mean([sol for sol, _ in self.memory], axis=0)
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb) * self.convergence_rate
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub).flatten()

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])
        if len(self.memory) > self.memory_size:
            self.memory.pop()

    def _dynamically_resize_memory(self):
        if self.successful_iterations > self.memory_size and len(self.memory) < 2 * self.memory_size:
            self.memory_size += 1
            self.successful_iterations = 0