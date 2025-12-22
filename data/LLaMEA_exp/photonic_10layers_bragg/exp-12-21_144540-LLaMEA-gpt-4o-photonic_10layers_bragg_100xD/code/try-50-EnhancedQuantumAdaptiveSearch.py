import numpy as np

class EnhancedQuantumAdaptiveSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(15, dim)  # Increased memory size for diversity
        self.memory = []  # Store solutions
        self.iteration = 0
        self.inertia_weight = 0.9  # Inertia weight for exploration/exploitation balance
        self.inertia_decay = 0.99  # Decay rate for inertia weight
        self.tunnel_rate = 0.05  # Parameter for stochastic tunneling

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')

        # Initial population with random solutions
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

            candidate_solution = self._quantum_explore(lb, ub, memory_solution)
            candidate_value = func(candidate_solution)

            if np.random.rand() < self.tunnel_rate:  # Apply stochastic tunneling
                tunneling_candidate = candidate_solution + np.random.normal(size=self.dim) * (ub - lb) * 0.1
                tunneling_candidate = np.clip(tunneling_candidate, lb, ub)
                tunneling_value = func(tunneling_candidate)
                if tunneling_value < candidate_value:
                    candidate_solution, candidate_value = tunneling_candidate, tunneling_value

            self.iteration += 1

            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            self._update_memory(candidate_solution, candidate_value)

            # Update inertia weight dynamically
            self.inertia_weight *= self.inertia_decay

        return best_solution

    def _select_from_memory(self):
        probabilities = np.array([1 / (i + 1) for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub, memory_solution):
        superposition = np.mean([sol for sol, _ in self.memory], axis=0)
        entanglement = self.inertia_weight * (memory_solution - superposition)
        candidate_solution = superposition + entanglement + np.random.uniform(-self.inertia_weight, self.inertia_weight, self.dim) * (ub - lb)
        return np.clip(candidate_solution, lb, ub).flatten()

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])
        if len(self.memory) > self.memory_size:
            self.memory.pop()