import numpy as np

class HybridQuantumAdaptiveSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)
        self.memory = []  # Store solutions
        self.iteration = 0
        self.adaptivity_rate = 0.1  # Initial adaptivity rate for exploration/exploitation balance
        self.phase_shift_rate = 0.1  # New parameter for phase shift diversification

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

            # Apply phase shift for diversification
            phase_shift_candidate = self._phase_shift(candidate_solution, lb, ub)
            phase_shift_value = func(phase_shift_candidate)

            # Apply gradient-based refinement for exploitation
            refined_candidate = self._gradient_refinement(phase_shift_candidate, func)
            refined_value = func(refined_candidate)

            # Select the best candidate from all transformations
            candidates = [(candidate_solution, candidate_value),
                          (phase_shift_candidate, phase_shift_value),
                          (refined_candidate, refined_value)]
            best_candidate, best_candidate_value = min(candidates, key=lambda x: x[1])

            self.iteration += 3

            if best_candidate_value < best_value:
                best_value = best_candidate_value
                best_solution = best_candidate

            self._update_memory(best_candidate, best_candidate_value)

        return best_solution

    def _select_from_memory(self):
        probabilities = np.array(
            [self.adaptivity_rate if i == 0 else np.random.rand() for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub):
        superposition = np.array([np.mean([sol for sol, _ in self.memory], axis=0)])
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub).flatten()

    def _phase_shift(self, solution, lb, ub):
        shift = np.random.uniform(-self.phase_shift_rate, self.phase_shift_rate, self.dim) * (ub - lb)
        shifted_solution = solution + shift
        return np.clip(shifted_solution, lb, ub)

    def _gradient_refinement(self, solution, func):
        epsilon = 1e-5
        gradient = np.zeros(self.dim)
        for i in range(self.dim):
            perturb = np.zeros(self.dim)
            perturb[i] = epsilon
            grad = (func(solution + perturb) - func(solution - perturb)) / (2 * epsilon)
            gradient[i] = grad
        refined_solution = solution - 0.01 * gradient
        return np.clip(refined_solution, func.bounds.lb, func.bounds.ub)

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])
        if len(self.memory) > self.memory_size:
            self.memory.pop()