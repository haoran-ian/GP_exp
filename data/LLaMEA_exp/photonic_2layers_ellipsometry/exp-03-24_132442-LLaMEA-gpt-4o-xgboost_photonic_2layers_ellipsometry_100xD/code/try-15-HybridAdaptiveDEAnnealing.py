import numpy as np

class HybridAdaptiveDEAnnealing:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        np.random.seed(42)

        # Initialize parameters
        T_initial = 1.0
        T_final = 0.001
        alpha = 0.9
        population_size = 10
        F = 0.8  # Differential weight
        CR = 0.9  # Crossover probability
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (population_size, self.dim))
        values = np.array([func(ind) for ind in population])
        best_idx = np.argmin(values)
        best_solution = np.copy(population[best_idx])
        best_value = values[best_idx]

        evaluations = population_size

        # Define dual annealing schedules
        schedule_A = lambda evals: T_initial * (T_final / T_initial) ** (evals / self.budget)
        schedule_B = lambda evals: T_initial * (0.5 + 0.5 * np.cos(np.pi * evals / self.budget))

        # Hybrid Adaptive DE and Simulated Annealing Loop
        while evaluations < self.budget:
            for i in range(population_size):
                # Differential Evolution mutant vector
                indices = [idx for idx in range(population_size) if idx != i]
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = population[a] + F * (population[b] - population[c])
                mutant_vector = np.clip(mutant_vector, func.bounds.lb, func.bounds.ub)

                # Crossover
                crossover_mask = np.random.rand(self.dim) < CR
                trial_vector = np.where(crossover_mask, mutant_vector, population[i])
                trial_value = func(trial_vector)
                evaluations += 1

                # Selection
                if trial_value < values[i]:
                    population[i] = trial_vector
                    values[i] = trial_value

                    # Update the best solution found
                    if trial_value < best_value:
                        best_solution = np.copy(trial_vector)
                        best_value = trial_value

            # Simulated Annealing: Adaptive Neighborhood Scaling
            for i in range(population_size):
                if evaluations < self.budget:
                    T = schedule_A(evaluations) if evaluations % 2 == 0 else schedule_B(evaluations)
                    scale = (func.bounds.ub - func.bounds.lb) * (1 - (evaluations / self.budget) ** (1 / alpha))
                    perturbation_factor = 1 + 0.1 * np.sin(np.pi * evaluations / self.budget)
                    perturbation = np.random.normal(0, scale / (5 * perturbation_factor), self.dim)
                    candidate_solution = population[i] + perturbation
                    candidate_solution = np.clip(candidate_solution, func.bounds.lb, func.bounds.ub)
                    candidate_value = func(candidate_solution)
                    evaluations += 1

                    # Metropolis criterion
                    if candidate_value < values[i] or np.random.rand() < np.exp((values[i] - candidate_value) / T):
                        population[i] = candidate_solution
                        values[i] = candidate_value

                        # Update the best solution found
                        if candidate_value < best_value:
                            best_solution = np.copy(candidate_solution)
                            best_value = candidate_value

        return best_solution, best_value