import numpy as np

class QuantumInspiredAdaptiveDiversitySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(20, dim)  # Increased memory size for diversity
        self.memory = []  # Store solutions
        self.iteration = 0
        self.adaptivity_rate = 0.1  # Initial adaptivity rate
        self.diversity_threshold = 0.05  # Diversity threshold for memory update

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

            # Quantum-inspired exploration with adaptive diversity preservation
            candidate_solution = self._quantum_explore(lb, ub)
            candidate_value = func(candidate_solution)

            self.iteration += 1

            # Update the best found solution
            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Update the memory with dynamic filtering and diversity check
            self._update_memory(candidate_solution, candidate_value)

        return best_solution

    def _select_from_memory(self):
        # Prefer better solutions with a probability based on adaptivity_rate
        probabilities = np.array(
            [self.adaptivity_rate if i == 0 else np.random.rand() for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub):
        # Quantum superposition to generate new candidate solution
        superposition = np.mean([sol for sol, _ in self.memory], axis=0)
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub)

    def _update_memory(self, solution, value):
        # Check for diversity and update memory
        if not any(np.linalg.norm(solution - sol) < self.diversity_threshold for sol, _ in self.memory):
            self.memory.append((solution, value))
            self.memory.sort(key=lambda x: x[1])  # Sort by value
            if len(self.memory) > self.memory_size:
                self.memory.pop()