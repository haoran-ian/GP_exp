import numpy as np

class EnhancedQuantumAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(15, dim)  # Increased memory size for better diversity
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
            # Select a solution from memory to exploit
            memory_solution, memory_value = self._select_from_memory()

            # Quantum-inspired exploration with dynamic entanglement control
            candidate_solution = self._quantum_explore(lb, ub)
            candidate_value = func(candidate_solution)

            self.iteration += 1

            # Update the best found solution
            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Update the memory
            self._update_memory(candidate_solution, candidate_value)

        return best_solution

    def _select_from_memory(self):
        # Prefer better solutions with probability based on adaptivity_rate
        scores = np.array([val for _, val in self.memory])
        probabilities = np.exp(-self.adaptivity_rate * (scores - scores.min()) / (scores.max() - scores.min() + 1e-9))
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub):
        # Quantum superposition to generate new candidate solution
        superposition = np.mean([sol for sol, _ in self.memory], axis=0)
        # Dynamic entanglement based on memory variance
        variance = np.var([val for _, val in self.memory])
        dynamic_entanglement = (1.0 / (1.0 + variance)) * (ub - lb) * np.random.uniform(-1, 1, self.dim)
        candidate_solution = superposition + dynamic_entanglement
        return np.clip(candidate_solution, lb, ub).flatten()

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])  # Sort by value (objective)
        if len(self.memory) > self.memory_size:
            self.memory.pop()