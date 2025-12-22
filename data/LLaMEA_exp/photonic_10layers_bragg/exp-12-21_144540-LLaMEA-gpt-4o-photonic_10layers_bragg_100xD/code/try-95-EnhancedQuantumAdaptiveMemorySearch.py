import numpy as np

class EnhancedQuantumAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = min(10, dim)  
        self.memory = []  
        self.iteration = 0
        self.adaptivity_rate = 0.1  
        self.phase_shift_rate = 0.1  
        self.gaussian_std = 0.1  

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

            gaussian_perturbed = self._gaussian_perturbation(candidate_solution, lb, ub)
            gaussian_value = func(gaussian_perturbed)

            if gaussian_value < candidate_value:
                candidate_solution, candidate_value = gaussian_perturbed, gaussian_value

            phase_shift_candidate = self._dynamic_phase_shift(candidate_solution, lb, ub, candidate_value, best_value)
            phase_shift_value = func(phase_shift_candidate)

            if phase_shift_value < candidate_value:
                candidate_solution, candidate_value = phase_shift_candidate, phase_shift_value

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

    def _quantum_explore(self, lb, ub):
        superposition = np.mean([sol for sol, _ in self.memory], axis=0)
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub)

    def _gaussian_perturbation(self, solution, lb, ub):
        perturbation = np.random.normal(0, self.gaussian_std, self.dim) * (ub - lb)
        perturbed_solution = solution + perturbation
        return np.clip(perturbed_solution, lb, ub)

    def _dynamic_phase_shift(self, solution, lb, ub, candidate_value, best_value):
        dynamic_shift_rate = self.phase_shift_rate * (1 + (best_value - candidate_value) / max(1e-9, best_value))
        shift = np.random.uniform(-dynamic_shift_rate, dynamic_shift_rate, self.dim) * (ub - lb)
        shifted_solution = solution + shift
        return np.clip(shifted_solution, lb, ub)

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])
        if len(self.memory) > self.memory_size:
            self.memory.pop()