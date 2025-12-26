import numpy as np

class AdaptiveQuantumInspiredMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)
        self.memory = []
        self.iteration = 0
        self.adaptivity_rate = 0.1
        self.learning_rate = 0.1  # Introducing a self-tuning learning rate

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

            # Self-tuning mechanism: adaptation based on progress
            progress = (best_value - memory_value) / (1 + abs(memory_value))
            self.adaptivity_rate = min(0.5, self.adaptivity_rate + 0.01 * progress)
            self.learning_rate = min(0.2, self.learning_rate + 0.005 * progress)

            candidate_solution = self._quantum_explore(lb, ub, memory_solution)
            candidate_value = func(candidate_solution)

            self.iteration += 1

            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            self._update_memory(candidate_solution, candidate_value)

        return best_solution

    def _select_from_memory(self):
        probabilities = np.array(
            [self.adaptivity_rate if i == 0 else np.random.rand() for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub, memory_solution):
        # Utilize the current best memory solution and learning rate for exploration
        superposition = np.array([memory_solution])
        entanglement = np.random.uniform(-self.learning_rate, self.learning_rate, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub).flatten()

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])
        if len(self.memory) > self.memory_size:
            self.memory.pop()