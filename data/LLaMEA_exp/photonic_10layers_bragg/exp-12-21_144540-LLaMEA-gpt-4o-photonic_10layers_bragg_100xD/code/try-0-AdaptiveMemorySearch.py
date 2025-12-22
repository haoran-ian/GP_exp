import numpy as np

class AdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)  # Fixed memory size for storing best solutions
        self.memory = []  # To store the best solutions found so far
        self.iteration = 0

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
        # Select a random memory entry
        idx = np.random.randint(len(self.memory))
        return self.memory[idx]

    def _explore(self, base_solution, lb, ub):
        # Generate a candidate solution by perturbing the base solution
        perturbation = np.random.uniform(-0.1, 0.1, self.dim) * (ub - lb)
        candidate_solution = base_solution + perturbation
        # Ensure the solution is within bounds
        return np.clip(candidate_solution, lb, ub)

    def _update_memory(self, solution, value):
        # Add the solution to memory, maintaining memory size
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])  # Sort memory by objective value
        if len(self.memory) > self.memory_size:
            self.memory.pop()  # Remove the worst solution if memory is full