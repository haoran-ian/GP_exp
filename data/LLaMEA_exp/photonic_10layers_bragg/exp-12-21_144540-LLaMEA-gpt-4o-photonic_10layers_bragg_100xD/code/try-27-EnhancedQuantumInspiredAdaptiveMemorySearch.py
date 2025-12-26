import numpy as np

class EnhancedQuantumInspiredAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(20, dim)  # Increased memory size for diversity
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
            # Select a diverse solution from memory to exploit
            memory_solution, memory_value = self._select_diverse_from_memory()

            # Adaptive explore phase
            candidate_solution = self._adaptive_quantum_explore(lb, ub, memory_solution)
            candidate_value = func(candidate_solution)

            self.iteration += 1

            # Update the best found solution
            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Update the memory
            self._update_memory(candidate_solution, candidate_value)

        return best_solution

    def _select_diverse_from_memory(self):
        # Enhance diversity by selecting solutions with different characteristics
        distances = np.array([np.linalg.norm(sol - self.memory[0][0]) for sol, _ in self.memory])
        diversity_probabilities = distances / distances.sum()
        idx = np.random.choice(len(self.memory), p=diversity_probabilities)
        return self.memory[idx]

    def _adaptive_quantum_explore(self, lb, ub, base_solution):
        # Adaptive quantum exploration based on a base solution
        superposition = np.array([base_solution])
        scaled_adaptivity = self.adaptivity_rate * (ub - lb)
        entanglement = np.random.uniform(-scaled_adaptivity, scaled_adaptivity, self.dim)
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub).flatten()

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])  # Sort by value (objective)
        if len(self.memory) > self.memory_size:
            self.memory.pop()