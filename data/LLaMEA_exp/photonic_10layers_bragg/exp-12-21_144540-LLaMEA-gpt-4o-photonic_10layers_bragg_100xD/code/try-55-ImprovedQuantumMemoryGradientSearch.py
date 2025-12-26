import numpy as np

class ImprovedQuantumMemoryGradientSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)
        self.memory = []  # Store solutions
        self.iteration = 0
        self.adaptivity_rate = 0.1
        self.tunnel_rate = 0.05
        self.gradient_impact = 0.1  # New parameter for gradient adjustment

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

            candidate_solution = self._quantum_explore(lb, ub, best_solution)
            candidate_value = func(candidate_solution)

            if np.random.rand() < self.tunnel_rate:
                tunneling_candidate = candidate_solution + np.random.normal(size=self.dim) * (ub - lb) * 0.1
                tunneling_candidate = np.clip(tunneling_candidate, lb, ub)
                tunneling_value = func(tunneling_candidate)
                candidate_solution, candidate_value = (tunneling_candidate, tunneling_value) if tunneling_value < candidate_value else (
                    candidate_solution, candidate_value)

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

    def _quantum_explore(self, lb, ub, best_solution):
        superposition = np.array([np.mean([sol for sol, _ in self.memory], axis=0)])
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb)
        gradient_adjustment = self.gradient_impact * (superposition - best_solution)
        candidate_solution = superposition + entanglement + gradient_adjustment
        return np.clip(candidate_solution, lb, ub).flatten()

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])
        if len(self.memory) > self.memory_size:
            self.memory.pop()