import numpy as np

class AdaptiveMemorySearchWithSelfTuning:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)  # Fixed memory size
        self.memory = []  # Store solutions
        self.iteration = 0
        self.exploration_rate = 0.1  # Initial exploration rate

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
            # Select a solution from memory
            memory_solution, memory_value = self._select_from_memory()

            # Adjust exploration rate dynamically
            improvement = max(0, best_value - memory_value)
            self.exploration_rate = min(0.5, self.exploration_rate + 0.01 * improvement)

            # Explore around the selected memory solution
            candidate_solution = self._explore(memory_solution, lb, ub)
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
        # Prefer better solutions with a probability based on exploration_rate
        probabilities = np.array([1.0 if i == 0 else self.exploration_rate for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _explore(self, base_solution, lb, ub):
        # Dynamic perturbation based on exploration_rate
        perturbation = np.random.uniform(-self.exploration_rate, self.exploration_rate, self.dim) * (ub - lb)
        candidate_solution = base_solution + perturbation
        return np.clip(candidate_solution, lb, ub)

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])  # Sort by value (objective)
        if len(self.memory) > self.memory_size:
            self.memory.pop()