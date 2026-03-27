import numpy as np

class HybridAdaptiveSimulatedAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)

        # Initialize parameters
        T_initial = 1.0
        T_final = 0.001
        alpha = 0.85
        current_solution = np.random.uniform(func.bounds.lb, func.bounds.ub, self.dim)
        current_value = func(current_solution)
        best_solution = np.copy(current_solution)
        best_value = current_value

        evaluations = 1

        # Define hybrid schedules with polynomial mutation
        schedule_A = lambda evals: T_initial * (T_final / T_initial) ** (evals / self.budget)
        schedule_B = lambda evals: T_initial * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget))

        # Polynomial mutation function
        def polynomial_mutation(solution, lbounds, ubounds, mutation_prob):
            mutated = np.copy(solution)
            for i in range(len(solution)):
                if np.random.rand() < mutation_prob:
                    u = np.random.rand()
                    delta = (2.0 * u) ** (1.0 / (1.0 + 0.5)) - 1.0 if u <= 0.5 else 1.0 - (2.0 * (1.0 - u)) ** (1.0 / (1.0 + 0.5))
                    mutated[i] += delta * (ubounds[i] - lbounds[i])
            return np.clip(mutated, lbounds, ubounds)

        # Hybrid Adaptive Simulated Annealing Loop
        while evaluations < self.budget:
            if evaluations % 4 == 0:
                T = schedule_A(evaluations)
                mutation_prob = 0.1  # Increased mutation probability when using schedule A
            else:
                T = schedule_B(evaluations)
                mutation_prob = 0.05  # Lower mutation probability when using schedule B

            # Adaptive Neighborhood Scaling with stochastic polynomial mutation
            scale = (func.bounds.ub - func.bounds.lb) * (1 - (evaluations / self.budget) ** (1 / alpha))
            perturbation = np.random.normal(0, scale / 5, self.dim)
            candidate_solution = current_solution + perturbation
            candidate_solution = polynomial_mutation(candidate_solution, func.bounds.lb, func.bounds.ub, mutation_prob)
            candidate_value = func(candidate_solution)
            evaluations += 1

            # Metropolis criterion
            if candidate_value < current_value or np.random.rand() < np.exp((current_value - candidate_value) / T):
                current_solution = candidate_solution
                current_value = candidate_value

                # Update the best solution found
                if current_value < best_value:
                    best_solution = np.copy(current_solution)
                    best_value = current_value

        return best_solution, best_value