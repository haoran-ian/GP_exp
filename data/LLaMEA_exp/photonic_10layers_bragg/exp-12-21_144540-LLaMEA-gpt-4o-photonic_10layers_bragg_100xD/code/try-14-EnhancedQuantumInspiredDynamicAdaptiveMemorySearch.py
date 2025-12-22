import numpy as np

class EnhancedQuantumInspiredDynamicAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)  # Initial memory size
        self.memory = []  # Store solutions and their fitness
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
            # Select a solution from memory to exploit
            memory_solution, memory_value = self._select_from_memory()

            # Update adaptivity rate based on difference from best
            self.adaptivity_rate = min(0.5, self.adaptivity_rate + 0.01 * (best_value - memory_value))

            # Quantum-inspired exploration with adaptive variance
            candidate_solution = self._adaptive_quantum_explore(lb, ub)
            candidate_value = func(candidate_solution)

            self.iteration += 1

            # Update the best found solution
            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Update the memory and adjust memory size dynamically
            self._update_memory(candidate_solution, candidate_value)
            self._adjust_memory_size()

        return best_solution

    def _select_from_memory(self):
        # Prefer better solutions with a probability based on adaptivity_rate
        probabilities = np.array([1.0 if i == 0 else self.adaptivity_rate for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _adaptive_quantum_explore(self, lb, ub):
        # Quantum superposition to generate new candidate solution
        superposition = np.mean([sol for sol, _ in self.memory], axis=0)
        variance = np.var([sol for sol, _ in self.memory], axis=0)
        adaptive_variance = np.maximum(variance, 0.1 * (ub - lb))
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * adaptive_variance
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub).flatten()

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])  # Sort by value (objective)
        if len(self.memory) > self.memory_size:
            self.memory.pop()

    def _adjust_memory_size(self):
        # Dynamically adjust memory size based on performance
        if len(self.memory) > 2 and self.memory[-1][1] > self.memory[-2][1]:
            self.memory_size = max(5, int(self.memory_size * 0.9))
        elif self.iteration > self.budget / 2 and len(self.memory) < self.memory_size:
            self.memory_size = min(20, self.memory_size + 1)