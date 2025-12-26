import numpy as np

class QuantumInspiredDynamicAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)  # Fixed memory size
        self.memory = []  # Store solutions
        self.iteration = 0
        self.exploration_factor = 0.1  # Initial exploration factor
        self.exploitation_factor = 0.1  # Initial exploitation factor
        self.performance_history = []  # Track recent performance

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

            # Update exploration and exploitation factors based on recent performance
            self._update_adaptivity(memory_value, best_value)

            # Quantum-inspired exploration with multi-level adaptivity
            candidate_solution = self._quantum_explore(lb, ub)
            candidate_value = func(candidate_solution)

            self.iteration += 1

            # Update the best found solution
            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Update the memory
            self._update_memory(candidate_solution, candidate_value)

            # Track recent performance
            self._track_performance(memory_value, candidate_value)

        return best_solution

    def _select_from_memory(self):
        # Prefer better solutions with a probability based on exploitation_factor
        probabilities = np.array(
            [np.random.rand() if i == 0 else self.exploitation_factor for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub):
        # Quantum superposition to generate new candidate solution
        superposition = np.array([np.mean([sol for sol, _ in self.memory], axis=0)])
        entanglement = np.random.uniform(-self.exploration_factor, self.exploration_factor, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub).flatten()

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])  # Sort by value (objective)
        if len(self.memory) > self.memory_size:
            self.memory.pop()

    def _update_adaptivity(self, memory_value, best_value):
        # Adjust exploration and exploitation factors based on performance
        performance_ratio = (best_value - memory_value) / max(abs(memory_value), 1e-10)
        self.exploration_factor = min(0.5, self.exploration_factor + 0.01 * performance_ratio)
        self.exploitation_factor = min(0.5, self.exploitation_factor + 0.01 * -performance_ratio)

    def _track_performance(self, memory_value, candidate_value):
        self.performance_history.append((memory_value, candidate_value))
        if len(self.performance_history) > 10:
            self.performance_history.pop(0)