import numpy as np

class EnhancedQuantumInspiredDynamicAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)
        self.memory = []
        self.iteration = 0
        self.adaptivity_rate = 0.1

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')

        # Initialize memory with random solutions
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
            self.adaptivity_rate = 0.1 + 0.2 * (best_value - memory_value) / (abs(best_value) + 1e-8)

            # Hybrid quantum-classical exploration
            if np.random.rand() < self.adaptivity_rate:
                candidate_solution = self._quantum_explore(lb, ub)
            else:
                candidate_solution = self._classical_explore(memory_solution, lb, ub)
            
            candidate_value = func(candidate_solution)
            self.iteration += 1

            # Update the best found solution
            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Update the memory
            self._update_memory(candidate_solution, candidate_value)

            # Dynamically adjust memory size based on progress
            self.memory_size = max(5, int((1 - best_value / (abs(best_value) + 1e-8)) * 10))
            self.memory = self.memory[:self.memory_size]

        return best_solution

    def _select_from_memory(self):
        probabilities = np.array([1.0 if i == 0 else self.adaptivity_rate for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub):
        superposition = np.mean([sol for sol, _ in self.memory], axis=0)
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub).flatten()

    def _classical_explore(self, base_solution, lb, ub):
        perturbation = np.random.normal(0, self.adaptivity_rate, self.dim) * (ub - lb)
        candidate_solution = base_solution + perturbation
        return np.clip(candidate_solution, lb, ub)

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])
        if len(self.memory) > self.memory_size:
            self.memory.pop()