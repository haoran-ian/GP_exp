import numpy as np

class OptimizedQuantumAdaptiveMemorySearch:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.memory_size = max(10, dim // 2)  # Dynamic memory size for better adaptability
        self.memory = []  # Store solutions
        self.iteration = 0
        self.adaptivity_rate = 0.1  # Initial adaptivity rate for exploration/exploitation balance
        self.phase_shift_rate = 0.1  # Phase shift rate for diversification
        self.compression_factor = 0.5  # Compression factor for solution memory

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

        # Main loop
        while self.iteration < self.budget:
            memory_solution, memory_value = self._select_from_memory()

            # Dynamically adjust adaptivity rate
            self.adaptivity_rate = min(0.5, self.adaptivity_rate + 0.02 * (best_value - memory_value))

            # Quantum-inspired exploration
            candidate_solution = self._quantum_explore(lb, ub)
            candidate_value = func(candidate_solution)

            # Enhanced phase shift for diversification
            phase_shift_candidate = self._enhanced_phase_shift(candidate_solution, best_solution, lb, ub)
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
        # Select with dynamically weighted probabilities
        probabilities = np.array([1.0 / (i + 1) for i in range(len(self.memory))])
        probabilities /= probabilities.sum()
        idx = np.random.choice(len(self.memory), p=probabilities)
        return self.memory[idx]

    def _quantum_explore(self, lb, ub):
        superposition = np.array([np.mean([sol for sol, _ in self.memory], axis=0)])
        entanglement = np.random.uniform(-self.adaptivity_rate, self.adaptivity_rate, self.dim) * (ub - lb)
        candidate_solution = superposition + entanglement
        return np.clip(candidate_solution, lb, ub).flatten()

    def _enhanced_phase_shift(self, solution, best_solution, lb, ub):
        # Use a blend of the current and best solution to enhance exploration
        blend = (solution + best_solution) / 2
        shift = np.random.uniform(-self.phase_shift_rate, self.phase_shift_rate, self.dim) * (ub - lb)
        shifted_solution = blend + shift
        return np.clip(shifted_solution, lb, ub)

    def _update_memory(self, solution, value):
        self.memory.append((solution, value))
        self.memory.sort(key=lambda x: x[1])
        # Compress memory to keep most relevant solutions
        if len(self.memory) > self.memory_size:
            self.memory = self.memory[:int(len(self.memory) * self.compression_factor)]