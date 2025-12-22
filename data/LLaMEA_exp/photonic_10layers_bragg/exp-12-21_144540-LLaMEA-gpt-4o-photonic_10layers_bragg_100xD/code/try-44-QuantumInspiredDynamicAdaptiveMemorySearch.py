import numpy as np

class QuantumInspiredDynamicAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)  # Fixed memory size
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

            # Update adaptivity rate based on difference from best
            self.adaptivity_rate = min(0.5, self.adaptivity_rate + 0.01 * (best_value - memory_value))

            # Quantum-inspired exploration with self-adaptive learning rate
            candidate_solution = self._quantum_explore(lb, ub)
            candidate_value = func(candidate_solution)

            self.iteration += 1

            # Intensification step with local search
            local_solution = self._local_search(candidate_solution, lb, ub, func)
            local_value = func(local_solution)
            self.iteration += 1

            # Update the best found solution
            if local_value < best_value:
                best_value = local_value
                best_solution = local_solution

            # Update the memory with dynamic filtering
            self._update_memory(local_solution, local_value)

        return best_solution

    def _select_from_memory(self):
        # Prefer better solutions with a probability based on adaptivity_rate
        probabilities = np.array(
            [self.adaptivity_rate if i == 0 else np.random.rand() for i in range(len(self.memory))])  # Line changed
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub):
        # Quantum superposition to generate new candidate solution
        superposition = np.array([np.mean([sol for sol, _ in self.memory], axis=0)])
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub).flatten()

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])  # Sort by value (objective)
        if len(self.memory) > self.memory_size:
            self.memory.pop()

    def _local_search(self, solution, lb, ub, func):
        # Conduct a simple local search around the candidate solution
        perturbation = np.random.uniform(-0.05, 0.05, self.dim) * (ub - lb)
        new_solution = solution + perturbation
        return np.clip(new_solution, lb, ub)