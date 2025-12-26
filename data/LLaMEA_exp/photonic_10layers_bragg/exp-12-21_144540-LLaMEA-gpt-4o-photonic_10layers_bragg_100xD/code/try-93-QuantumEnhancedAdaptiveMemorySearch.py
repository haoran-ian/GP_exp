import numpy as np

class QuantumEnhancedAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)
        self.memory = []
        self.iteration = 0
        self.adaptivity_rate = 0.1
        self.phase_shift_rate = 0.1
        self.learning_rate = 0.01  # Adds learning rate for adaptive updates

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

            # Self-tuning adaptivity rate
            self.adaptivity_rate = min(0.5, max(0.01, self.adaptivity_rate + self.learning_rate * (best_value - memory_value)))

            candidate_solution = self._quantum_explore(lb, ub)
            candidate_value = func(candidate_solution)

            phase_shift_candidate = self._phase_shift(candidate_solution, lb, ub)
            phase_shift_value = func(phase_shift_candidate)
            candidate_solution, candidate_value = (phase_shift_candidate, phase_shift_value) if phase_shift_value < candidate_value else (candidate_solution, candidate_value)

            self.iteration += 1

            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            self._update_memory(candidate_solution, candidate_value)

        return best_solution

    def _select_from_memory(self):
        # Memory selection with adaptive probabilities
        scores = np.array([v for _, v in self.memory])
        probabilities = np.exp(-scores / (self.adaptivity_rate + 1e-9))
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub):
        superposition = np.mean([sol for sol, _ in self.memory], axis=0)
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub)

    def _phase_shift(self, solution, lb, ub):
        shift = np.random.uniform(-self.phase_shift_rate, self.phase_shift_rate, self.dim) * (ub - lb)
        shifted_solution = solution + shift
        return np.clip(shifted_solution, lb, ub)

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[:self.memory_size]