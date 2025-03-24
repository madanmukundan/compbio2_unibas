from Bacterium import Bacterium
import queue
from scipy import stats
import concurrent.futures
import time
import numpy as np
import matplotlib.pyplot as plt


class LuriaDelbruckSimulation:
    def __init__(self, start_pop_size=10,
                 end_pop_size=1000,
                 division_distribution="poisson",
                 mutation_rate=1e-2,
                 lifespan=20,
                 lifespan_distribution="fixed",
                 lifespan_mean=20,
                 lifespan_std=5,
                 lifespan_lambda=20,
                 division_mean=10,
                 division_std=2,
                 division_lambda=10,
                 base_survival_prob=0.001,
                 mutation_survival_boost=10.0):

        # Passed up to Bacterium Constructor
        self.start_pop_size = start_pop_size
        self.end_pop_size = end_pop_size
        self.division_distribution = division_distribution
        self.mutation_rate = mutation_rate
        self.lifespan = lifespan
        self.lifespan_distribution = lifespan_distribution
        self.lifespan_mean = lifespan_mean
        self.lifespan_std = lifespan_std
        self.lifespan_lambda = lifespan_lambda
        self.base_survival_prob = base_survival_prob
        self.mutation_survival_boost = mutation_survival_boost

        # Gaussian distribution paramenters
        self.division_mean = division_mean
        self.division_std = division_std

        # Poisson distribution parameters
        self.division_lambda = division_lambda

        # Initialize population and event queue
        self.population = []
        self.event_queue = queue.PriorityQueue()  # use time step for scheduling priority
        self.current_time = 0  # curent time step as int
        self.next_event_id = 0  # ensure fifo for events with same time

        # Statistics
        self.population_history = []  # Size of population at each event
        self.mutant_history = []  # Mutant count at each event
        self.total_divisions = 0
        self.total_mutations = 0

    def initialize_population(self):
        """Create the initial bacterial population."""
        self.population = []
        for _ in range(self.start_pop_size):
            bacterium = Bacterium(
                phenotype="sensitive",
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
                base_survival_prob=self.base_survival_prob,
                mutation_survival_boost=self.mutation_survival_boost
            )
            self.population.append(bacterium)

            # Schedule division for this bacterium
            self.schedule_event(bacterium.time_to_division, ("division", len(self.population) - 1))
            
            # Schedule death for this bacterium
            self.schedule_event(bacterium.time_to_death, ("death", len(self.population) - 1))

    def schedule_event(self, time_delta, event_data):
        """Add an event to the queue to occur after time_delta."""
        event_time = self.current_time + time_delta
        self.event_queue.put((event_time, self.next_event_id, event_data))
        self.next_event_id += 1

    def run_simulation(self, callback=None) -> "dict":
        self.initialize_population()

        # Run the simulation
        while len(self.population) < self.end_pop_size and not self.event_queue.empty():

            # Get next event in event queue, toss event id
            event_time, _, (event_type, bacterium_idx) = self.event_queue.get()
            self.current_time = event_time

            # Skip if the bacterium has died or index is invalid
            if bacterium_idx >= len(self.population) or self.population[bacterium_idx] is None:
                continue

            bacterium = self.population[bacterium_idx]

            if event_type == "division":
                # Perform division
                new_bacterium, mutation_occurred = bacterium.divide()

                if mutation_occurred:
                    self.total_mutations += 1

                self.total_divisions += 1

                # Add the new bacterium to the population
                self.population.append(new_bacterium)

                # Schedule next division event for parent
                self.schedule_event(bacterium.time_to_division, ("division", bacterium_idx))
                
                # Schedule division and death events for offspring
                self.schedule_event(new_bacterium.time_to_division, ("division", len(self.population) - 1))
                self.schedule_event(new_bacterium.time_to_death, ("death", len(self.population) - 1))
            
            elif event_type == "death":
                # Bacterium dies (remove from population)
                self.population[bacterium_idx] = None

            # Record current statistics
            active_population = [b for b in self.population if b is not None]
            mutant_count = sum(1 for b in active_population if b.mutant_status > 0)

            self.population_history.append(len(active_population))
            self.mutant_history.append(mutant_count)

            # For GUI call callback with progress updates if provided
            if callback and len(self.population_history) % 100 == 0:
                callback(len(active_population), mutant_count)

        # Now evaluate survival for all living bacteria at the end of simulation
        # Remove none
        active_population = [b for b in self.population if b is not None]
        
        # Determine survival for each bacterium
        for bacterium in active_population:
            bacterium.determine_survival()
        
        # Count survivors and mutants among the survivors
        surviving_bacteria = [b for b in active_population if b.survived]
        mutant_count = sum(1 for b in active_population if b.mutant_status > 0)
        surviving_mutant_count = sum(1 for b in surviving_bacteria if b.mutant_status > 0)
        
        # Calculate proportions
        total_count = len(active_population)
        survival_proportion = len(surviving_bacteria) / total_count if total_count > 0 else 0
        mutant_proportion = mutant_count / total_count if total_count > 0 else 0
        surviving_mutant_proportion = surviving_mutant_count / len(surviving_bacteria) if surviving_bacteria else 0

        return {
            "population_size": total_count,
            "mutant_count": mutant_count,
            "mutant_proportion": mutant_proportion,
            "surviving_count": len(surviving_bacteria),
            "survival_proportion": survival_proportion,
            "surviving_mutant_count": surviving_mutant_count,
            "surviving_mutant_proportion": surviving_mutant_proportion,
            "total_divisions": self.total_divisions,
            "total_mutations": self.total_mutations,
            "population_history": self.population_history,
            "mutant_history": self.mutant_history,
            "final_time": self.current_time
        }


class LDSimulationRunner:
    """Class to manage multiple simulation runs."""

    def __init__(self):
        """Initialize the simulation runner."""
        self.results = []
        self.survival_results = []
        self.surviving_mutant_results = []
        self.raw_results = []

    def run_single_simulation(self, params, callback=None) -> "dict":
        """Run Single Luria Delbruck Simulation"""
        sim = LuriaDelbruckSimulation(
            start_pop_size=params["start_pop_size"],
            end_pop_size=params["end_pop_size"],
            division_distribution=params["division_distribution"],
            mutation_rate=params["mutation_rate"],
            lifespan=params["lifespan"],
            lifespan_distribution=params["lifespan_distribution"],
            lifespan_mean=params["lifespan_mean"],
            lifespan_std=params["lifespan_std"],
            lifespan_lambda=params["lifespan_lambda"],
            division_mean=params["division_mean"],
            division_std=params["division_std"],
            division_lambda=params["division_lambda"],
            base_survival_prob=params.get("base_survival_prob", 0.001),
            mutation_survival_boost=params.get("mutation_survival_boost", 10.0)
        )

        result = sim.run_simulation(callback)
        return result

    def run_sequential_simulations(self, params, num_runs, progress_callback=None) -> "list":
        self.results = []
        self.survival_results = []
        self.surviving_mutant_results = []
        self.raw_results = []

        for i in range(num_runs):
            result = self.run_single_simulation(params)
            self.results.append(result["mutant_proportion"])
            self.survival_results.append(result["survival_proportion"])
            self.surviving_mutant_results.append(result["surviving_mutant_proportion"])
            self.raw_results.append(result)

            if progress_callback:
                progress_callback(i + 1, num_runs)

        return self.results

    def _parallel_sim_worker(self, params, sim_index):
        """Worker function for parallel simulation"""
        try:
            result = self.run_single_simulation(params)
            return sim_index, result
        except Exception as e:
            return sim_index, f"Error: {str(e)}"

    def run_parallel_simulations(self, params, num_runs, progress_callback=None) -> "list":
        """Run multiple simulations in parallel with ThreadPoolExecutor"""
        self.results = [None] * num_runs
        self.survival_results = [None] * num_runs
        self.surviving_mutant_results = [None] * num_runs
        self.raw_results = [None] * num_runs

        # Use ThreadPoolExecutor for parallel execution
        with concurrent.futures.ThreadPoolExecutor() as executor:
            # Submit all tasks
            future_to_index = {
                executor.submit(self._parallel_sim_worker, params, i): i 
                for i in range(num_runs)
            }
            
            # Process results as they complete
            completed = 0
            for future in concurrent.futures.as_completed(future_to_index):
                sim_index, result = future.result()
                if isinstance(result, dict):
                    self.results[sim_index] = result["mutant_proportion"]
                    self.survival_results[sim_index] = result["survival_proportion"]
                    self.surviving_mutant_results[sim_index] = result["surviving_mutant_proportion"]
                    self.raw_results[sim_index] = result
                
                completed += 1
                if progress_callback:
                    progress_callback(completed, num_runs)

        return self.results

    def plot_results(self, ax=None):
        """Return plot of histogram and kde of mutant proportions and survival rates"""
        if ax is None:
            fig, axs = plt.subplots(2, 1, figsize=(10, 12))
            ax_mutants = axs[0]
            ax_survival = axs[1]
        else:
            # If single axis provided, create a new figure for the second plot
            fig = ax.figure
            ax_mutants = ax
            fig_survival, ax_survival = plt.subplots(figsize=(10, 6))

        if not self.results:
            ax_mutants.text(0.5, 0.5, "No simulation results to display",
                   ha="center", va="center", transform=ax_mutants.transAxes)
            if ax_survival:
                ax_survival.text(0.5, 0.5, "No simulation results to display",
                       ha="center", va="center", transform=ax_survival.transAxes)
            return fig

        # Plot mutant proportion histogram
        ax_mutants.hist(self.results, bins=20, density=True, alpha=0.6, color="skyblue")
        ax_mutants.set_xlabel("Proportion of Mutants in Total Population")
        ax_mutants.set_ylabel("Density")
        ax_mutants.set_title("Luria-Delbrück Experiment: Distribution of Mutant Proportions")

        # Plot survival proportion histogram
        ax_survival.hist(self.survival_results, bins=20, density=True, alpha=0.6, color="lightgreen")
        ax_survival.hist(self.surviving_mutant_results, bins=20, density=True, alpha=0.6, color="salmon")
        ax_survival.set_xlabel("Proportion")
        ax_survival.set_ylabel("Density")
        ax_survival.set_title("Survival Rates: Total Survival (green) and Mutants Among Survivors (red)")
        ax_survival.legend(["Total Survival Rate", "Mutant Proportion Among Survivors"])

        # Add KDE if we have enough data points
        if len(self.results) >= 5:
            try:
                # KDE for mutant proportion
                if any(x > 0 for x in self.results):
                    kde_mutants = stats.gaussian_kde(self.results)
                    x_mutants = np.linspace(0, max(self.results) * 1.1, 1000)
                    ax_mutants.plot(x_mutants, kde_mutants(x_mutants), 'r-', lw=2)
                
                # KDE for survival proportion
                if any(x > 0 for x in self.survival_results):
                    kde_survival = stats.gaussian_kde(self.survival_results)
                    x_survival = np.linspace(0, max(self.survival_results) * 1.1, 1000)
                    ax_survival.plot(x_survival, kde_survival(x_survival), 'g-', lw=2)
                
                # KDE for surviving mutants proportion
                if any(x > 0 for x in self.surviving_mutant_results):
                    kde_surv_mutants = stats.gaussian_kde(self.surviving_mutant_results)
                    x_surv_mutants = np.linspace(0, max(self.surviving_mutant_results) * 1.1, 1000)
                    ax_survival.plot(x_surv_mutants, kde_surv_mutants(x_surv_mutants), 'r-', lw=2)
            except Exception:
                # KDE might fail
                pass

        plt.tight_layout()
        return fig
