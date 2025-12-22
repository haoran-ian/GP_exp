import numpy as np

class QuantumInspiredDynamicAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)  # Fixed memory size
        self.memory = []  # Store solutions
        self.iteration = 0
        self.adaptivity_rate = 0.1  # Initial adaptivity rate for exploration/exploitation balance
        self.scale_factors = [0.5, 1.0, 2.0]  # Different scales for exploration

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
            self.adaptivity_rate = min(0.5, max(0.01, self.adaptivity_rate + 0.01 * (best_value - memory_value)))

            # Multi-Scale Quantum-inspired exploration
            candidate_solution = self._multi_scale_explore(lb, ub)
            candidate_value = func(candidate_solution)

            self.iteration += 1

            # Update the best found solution
            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Recalibrate memory dynamically
            self._update_memory(candidate_solution, candidate_value)

        return best_solution

    def _select_from_memory(self):
        # Prefer better solutions with a probability based on adaptivity_rate
        probabilities = np.array(
            [self.adaptivity_rate if i == 0 else np.random.rand() for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _multi_scale_explore(self, lb, ub):
        # Quantum superposition to generate new candidate solution
        superposition = np.array([np.mean([sol for sol, _ in self.memory], axis=0)])
        scale = np.random.choice(self.scale_factors)
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb) * scale
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub).flatten()

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory = sorted(self.memory, key=lambda x: x[1])[:self.memory_size]  # Sort and keep best