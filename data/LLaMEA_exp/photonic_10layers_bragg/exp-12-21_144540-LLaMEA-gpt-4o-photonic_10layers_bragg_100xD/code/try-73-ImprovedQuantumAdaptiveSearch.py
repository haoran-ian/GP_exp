import numpy as np

class ImprovedQuantumAdaptiveSearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(20, dim * 2)  # Increased population size for better diversity
        self.population = []  # Store solutions
        self.iteration = 0
        self.adaptivity_rate = 0.1  # Initial adaptivity rate for exploration/exploitation balance
        self.phase_shift_rate = 0.2  # Increased phase shift rate for more diversification

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')

        # Initial population
        for _ in range(self.population_size):
            solution = np.random.uniform(lb, ub, self.dim)
            value = func(solution)
            self._update_population(solution, value)
            if value < best_value:
                best_value = value
                best_solution = solution

        self.iteration += self.population_size

        while self.iteration < self.budget:
            population_solution, population_value = self._select_from_population()

            # Dynamic adaptivity rate adjustment
            self.adaptivity_rate = min(0.7, max(0.1, self.adaptivity_rate + 0.01 * (best_value - population_value) / (abs(best_value) + 1e-8)))

            candidate_solution = self._quantum_explore(lb, ub)
            candidate_value = func(candidate_solution)

            # Apply phase shift for diversification
            phase_shift_candidate = self._phase_shift(candidate_solution, lb, ub)
            phase_shift_value = func(phase_shift_candidate)
            candidate_solution, candidate_value = (phase_shift_candidate, phase_shift_value) if phase_shift_value < candidate_value else (candidate_solution, candidate_value)

            self.iteration += 1

            if candidate_value < best_value:
                best_value = candidate_value
                best_solution = candidate_solution

            self._update_population(candidate_solution, candidate_value)

        return best_solution

    def _select_from_population(self):
        probabilities = np.array(
            [self.adaptivity_rate if i == 0 else np.random.rand() for i in range(len(self.population))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.population), p=probabilities)
        return self.population[idx]

    def _quantum_explore(self, lb, ub):
        superposition = np.mean([sol for sol, _ in self.population], axis=0)
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub)

    def _phase_shift(self, solution, lb, ub):
        shift = np.random.uniform(-self.phase_shift_rate, self.phase_shift_rate, self.dim) * (ub - lb)
        shifted_solution = solution + shift
        return np.clip(shifted_solution, lb, ub)

    def _update_population(self, solution, value):
        self.population.append((solution, value))
        self.population.sort(key=lambda x: x[1])
        if len(self.population) > self.population_size:
            self.population.pop()