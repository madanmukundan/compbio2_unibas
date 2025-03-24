import random
import numpy as np


class Bacterium:

    def __init__(self, phenotype="sensitive",
                 mutation_rate=1e-2,
                 lifespan=20,
                 lifespan_distribution="fixed",
                 lifespan_mean=20,
                 lifespan_std=5,
                 lifespan_lambda=20,
                 division_distribution="poisson",
                 division_mean=10,
                 division_std=2,
                 division_lambda=10,
                 mutant_status=0,
                 base_survival_prob=0.001,  # Base survival probability (0.1%)
                 mutation_survival_boost=10.0):  # How much mutation improves survival

        self.phenotype = phenotype
        self.mutation_rate = mutation_rate
        self.lifespan = lifespan
        self.mutant_status = mutant_status
        
        # Survival parameters
        self.base_survival_prob = base_survival_prob
        self.mutation_survival_boost = mutation_survival_boost
        self.survival_determined = False
        self.survived = False
        
        # Division parameters
        self.division_distribution = division_distribution
        self.division_mean = division_mean
        self.division_std = division_std
        self.division_lambda = division_lambda
        
        # Lifespan parameters
        self.lifespan_distribution = lifespan_distribution
        self.lifespan_mean = lifespan_mean
        self.lifespan_std = lifespan_std
        self.lifespan_lambda = lifespan_lambda

        # Set initial time to division based on the chosen distribution
        self.time_to_division = self._get_division_time()
        
        # Set time to death based on lifespan distribution
        self.time_to_death = self._get_death_time()

    def _get_division_time(self):
        """Generate time to next division based on the chosen distribution."""
        if self.division_distribution == "gaussian":
            # Ensure positive value with floor at 1
            return max(1, int(np.random.normal(self.division_mean, self.division_std)))
        elif self.division_distribution == "poisson":
            return max(1, np.random.poisson(self.division_lambda))
        else:
            # Default to fixed value if distribution not recognized
            return 10
            
    def _get_death_time(self):
        """Generate time to death based on the chosen distribution."""
        if self.lifespan_distribution == "gaussian":
            # Ensure positive value with floor at 1
            return max(1, int(np.random.normal(self.lifespan_mean, self.lifespan_std)))
        elif self.lifespan_distribution == "poisson":
            return max(1, np.random.poisson(self.lifespan_lambda))
        else:
            # Default to fixed value
            return self.lifespan
            
    def determine_survival(self):
        """Determine if the bacterium survives the end-of-simulation challenge"""
        if self.survival_determined:
            return self.survived
            
        # Mutants have an increased survival probability
        if self.mutant_status > 0:
            # If default survival is zero, interprete mutant status (1) = survives
            if self.base_survival_prob == 0:
                survival_prob = 1
            else:
                # Otherwise probability to survive is determined based on mutation rate
                survival_prob = self.base_survival_prob * self.mutation_survival_boost
        else:
            survival_prob = self.base_survival_prob
            
        # Cap survival probability at 1.0 (100%)
        survival_prob = min(1.0, survival_prob)
        
        # Determine survival
        self.survived = random.random() < survival_prob
        self.survival_determined = True
        
        return self.survived

    def divide(self):
        """ Handle cell division, potentially with mutation. """
        # Get new division time
        self.time_to_division = self._get_division_time()

        # Create offspring with same properties as parent
        new_bacterium = Bacterium(
            phenotype=self.phenotype,
            mutation_rate=self.mutation_rate,
            lifespan=self.lifespan,
            lifespan_distribution=self.lifespan_distribution,
            lifespan_mean=self.lifespan_mean,
            lifespan_std=self.lifespan_std,
            lifespan_lambda=self.lifespan_lambda,
            division_distribution=self.division_distribution,
            division_mean=self.division_mean,
            division_std=self.division_std,
            division_lambda=self.division_lambda,
            mutant_status=self.mutant_status,
            base_survival_prob=self.base_survival_prob,
            mutation_survival_boost=self.mutation_survival_boost
        )

        # Parent mutation at division
        parent_mutation_occurred = random.random() < self.mutation_rate
        if parent_mutation_occurred and self.phenotype == "sensitive":
            self.phenotype = "resistant"
            self.mutant_status = 1

        # Child mutation at division
        offspring_mutation_occurred = random.random() < self.mutation_rate
        if offspring_mutation_occurred and new_bacterium.phenotype == "sensitive":
            new_bacterium.phenotype = "resistant"
            new_bacterium.mutant_status = 1

        return new_bacterium, parent_mutation_occurred
