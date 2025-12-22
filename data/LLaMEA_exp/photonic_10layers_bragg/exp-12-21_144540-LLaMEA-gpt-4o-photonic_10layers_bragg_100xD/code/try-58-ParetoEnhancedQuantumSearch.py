import numpy as np

class ParetoEnhancedQuantumSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)
        self.memory = []
        self.iteration = 0
        self.adaptivity_rate = 0.1
        self.tunnel_rate = 0.05

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

            candidate_solution = self._quantum_explore(lb, ub)
            candidate_value = func(candidate_solution)

            if np.random.rand() < self.tunnel_rate:
                tunneling_candidate = candidate_solution + np.random.normal(size=self.dim) * (ub - lb) * 0.1
                tunneling_candidate = np.clip(tunneling_candidate, lb, ub)
                tunneling_value = func(tunneling_candidate)
                candidate_solution, candidate_value = (tunneling_candidate, tunneling_value) if tunneling_value < candidate_value else (candidate_solution, candidate_value)

            self.iteration += 1

            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            self._update_memory(candidate_solution, candidate_value)

        return best_solution

    def _select_from_memory(self):
        pareto_front = self._pareto_filter(self.memory)
        return min(pareto_front, key=lambda x: x[1])

    def _quantum_explore(self, lb, ub):
        superposition = np.mean([sol for sol, _ in self.memory], axis=0)
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub)

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        if len(self.memory) > self.memory_size:
            self.memory = self._pareto_filter(self.memory)

    def _pareto_filter(self, memories):
        memories = sorted(memories, key=lambda x: x[1])
        pareto_front = [memories[0]]
        for sol, val in memories[1:]:
            if val < pareto_front[-1][1]:
                pareto_front.append((sol, val))
        return pareto_front