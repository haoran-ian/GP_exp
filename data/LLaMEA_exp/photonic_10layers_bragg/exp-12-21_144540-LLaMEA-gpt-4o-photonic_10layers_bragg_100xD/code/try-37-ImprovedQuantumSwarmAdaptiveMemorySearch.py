import numpy as np

class ImprovedQuantumSwarmAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)  # Fixed memory size
        self.memory = []  # Store solutions
        self.iteration = 0
        self.adaptivity_rate = 0.1  # Initial adaptivity rate for exploration/exploitation balance
        self.confidence_level = 0.5  # Confidence level for resource allocation

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

            # Quantum-inspired swarm exploration with adaptive learning rate
            candidate_solution = self._quantum_swarm_explore(lb, ub)
            candidate_value = func(candidate_solution)

            self.iteration += 1

            # Update confidence level for resource allocation
            self.confidence_level = max(0.1, self.confidence_level * (1 + 0.01 * (best_value - candidate_value)))

            # Allocate resources based on confidence
            if self.confidence_level > 0.5:
                candidate_solution = self._refine_solution(candidate_solution, lb, ub)

            # Update the best found solution
            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            # Update the memory with dynamic filtering
            self._update_memory(candidate_solution, candidate_value)

        return best_solution

    def _select_from_memory(self):
        # Prefer better solutions with a probability based on adaptivity_rate
        probabilities = np.array(
            [self.adaptivity_rate if i == 0 else np.random.rand() for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_swarm_explore(self, lb, ub):
        # Quantum superposition with swarm intelligence to generate new candidate solution
        superposition = np.mean([sol for sol, _ in self.memory], axis=0)
        entanglement_factor = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim)
        swarm_influence = np.random.normal(0, 0.1, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement_factor + swarm_influence
        return np.clip(candidate_solution, lb, ub)

    def _refine_solution(self, solution, lb, ub):
        # Small refinement step to improve solution quality
        noise = np.random.normal(0, 0.01, self.dim) * (ub - lb)
        refined_solution = solution + noise
        return np.clip(refined_solution, lb, ub)

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])  # Sort by value (objective)
        if len(self.memory) > self.memory_size:
            self.memory.pop()