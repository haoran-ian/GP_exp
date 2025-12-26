import numpy as np

class QuantumGuidedAdaptiveMultiPhaseOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)
        self.memory = []
        self.iteration = 0
        self.adaptivity_rate = 0.1
        self.phase_length = 10  # Number of iterations before switching phases
        self.current_phase = 0  # Start with exploration phase

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
            if self.iteration % self.phase_length == 0:
                self.current_phase = (self.current_phase + 1) % 2  # Alternate between phases

            if self.current_phase == 0:
                # Exploration phase
                candidate_solution = self._quantum_explore(lb, ub)
            else:
                # Exploitation phase
                candidate_solution, _ = self._select_from_memory()

            candidate_value = func(candidate_solution)
            self.iteration += 1

            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Adjust adaptivity rate with a sinusoidal function for dynamic transitions
            self.adaptivity_rate = 0.5 * (1 + np.sin(self.iteration * np.pi / self.phase_length))

            self._update_memory(candidate_solution, candidate_value)

        return best_solution

    def _select_from_memory(self):
        probabilities = np.array([self.adaptivity_rate if i == 0 else np.random.rand() for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub):
        superposition = np.array([np.mean([sol for sol, _ in self.memory], axis=0)])
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub).flatten()

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])
        if len(self.memory) > self.memory_size:
            self.memory.pop()