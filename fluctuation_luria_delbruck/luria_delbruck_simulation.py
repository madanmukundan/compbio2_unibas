import tkinter as tk
from tkinter import ttk, messagebox
import customtkinter as ctk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import queue
import random
import time
import threading
import multiprocessing as mp
from scipy import stats
import pandas as pd
import os

# Set appearance mode and color theme for customtkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class Bacterium:
    """Class representing a single bacterium in the simulation."""

    def __init__(self, phenotype="sensitive", mutation_rate=1e-7, lifespan=20,
                 division_distribution="gaussian", division_mean=10, division_std=2,
                 division_lambda=10, mutant_status=0):
        """
        Initialize a bacterium with given properties.

        Args:
            phenotype (str): "sensitive" or "resistant"
            mutation_rate (float): Probability of mutation per division
            lifespan (int): Number of divisions until death
            division_distribution (str): "gaussian" or "poisson"
            division_mean (float): Mean for Gaussian distribution
            division_std (float): Standard deviation for Gaussian distribution
            division_lambda (float): Lambda parameter for Poisson distribution
            mutant_status (int): 0 for non-mutant, 1 for mutant
        """
        self.age = 0
        self.phenotype = phenotype
        self.mutation_rate = mutation_rate
        self.lifespan = lifespan
        self.mutant_status = mutant_status
        self.division_distribution = division_distribution
        self.division_mean = division_mean
        self.division_std = division_std
        self.division_lambda = division_lambda

        # Set initial time to division based on the chosen distribution
        self.time_to_division = self._get_division_time()

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

    def tick(self):
        """Process one time step for this bacterium."""
        self.time_to_division -= 1
        return self.time_to_division <= 0

    def divide(self):
        """
        Handle cell division, potentially with mutation.

        Returns:
            tuple: (new_bacterium, has_parent_mutated)
        """
        self.age += 1

        # Reset division timer
        self.time_to_division = self._get_division_time()

        # Check for mutation in parent
        parent_mutation_occurred = random.random() < self.mutation_rate

        if parent_mutation_occurred and self.phenotype == "sensitive":
            self.phenotype = "resistant"
            self.mutant_status = 1

        # Create offspring with same properties as parent
        new_bacterium = Bacterium(
            phenotype=self.phenotype,
            mutation_rate=self.mutation_rate,
            lifespan=self.lifespan,
            division_distribution=self.division_distribution,
            division_mean=self.division_mean,
            division_std=self.division_std,
            division_lambda=self.division_lambda,
            mutant_status=self.mutant_status
        )

        # Check for mutation in offspring
        offspring_mutation_occurred = random.random() < self.mutation_rate

        if offspring_mutation_occurred and new_bacterium.phenotype == "sensitive":
            new_bacterium.phenotype = "resistant"
            new_bacterium.mutant_status = 1

        return new_bacterium, parent_mutation_occurred

    def should_die(self):
        """Check if bacterium has reached its lifespan."""
        return self.age >= self.lifespan


class LuriaDelbruckSimulation:
    """Class implementing the Luria-Delbrück experiment simulation."""

    def __init__(self, start_pop_size=10, end_pop_size=1000,
                 division_distribution="gaussian", mutation_rate=1e-6,
                 lifespan=20, division_mean=10, division_std=2,
                 division_lambda=10, antibiotic_challenge=False):
        """
        Initialize the simulation with given parameters.

        Args:
            start_pop_size (int): Initial population size
            end_pop_size (int): Target population size
            division_distribution (str): "gaussian" or "poisson"
            mutation_rate (float): Mutation rate per division
            lifespan (int): Bacterial lifespan (in divisions)
            division_mean (float): Mean for Gaussian distribution
            division_std (float): Standard deviation for Gaussian distribution
            division_lambda (float): Lambda parameter for Poisson distribution
            antibiotic_challenge (bool): Whether to apply antibiotic selection
        """
        self.start_pop_size = start_pop_size
        self.end_pop_size = end_pop_size
        self.division_distribution = division_distribution
        self.mutation_rate = mutation_rate
        self.lifespan = lifespan
        self.division_mean = division_mean
        self.division_std = division_std
        self.division_lambda = division_lambda
        self.antibiotic_challenge = antibiotic_challenge

        # Initialize population and event queue
        self.population = []
        self.event_queue = queue.PriorityQueue()
        self.current_time = 0
        self.next_event_id = 0  # To ensure FIFO behavior for events with same time

        # Statistics
        self.population_history = []
        self.mutant_history = []
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
                division_distribution=self.division_distribution,
                division_mean=self.division_mean,
                division_std=self.division_std,
                division_lambda=self.division_lambda
            )
            self.population.append(bacterium)

            # Schedule first division for this bacterium
            self.schedule_event(bacterium.time_to_division, ("division", len(self.population) - 1))

    def schedule_event(self, time_delta, event_data):
        """Add an event to the queue to occur after time_delta."""
        event_time = self.current_time + time_delta
        self.event_queue.put((event_time, self.next_event_id, event_data))
        self.next_event_id += 1

    def run_simulation(self, callback=None):
        """
        Run the simulation until the target population size is reached.

        Args:
            callback: Optional function to call periodically with progress updates

        Returns:
            dict: Simulation results
        """
        self.initialize_population()

        # Run the simulation
        while len(self.population) < self.end_pop_size and not self.event_queue.empty():
            # Process the next event
            event_time, _, (event_type, bacterium_idx) = self.event_queue.get()
            self.current_time = event_time

            # Skip if the bacterium has died or index is invalid
            if bacterium_idx >= len(self.population) or self.population[bacterium_idx] is None:
                continue

            bacterium = self.population[bacterium_idx]

            if event_type == "division":
                if not bacterium.should_die():
                    # Perform division
                    new_bacterium, mutation_occurred = bacterium.divide()

                    if mutation_occurred:
                        self.total_mutations += 1

                    self.total_divisions += 1

                    # Add the new bacterium to the population
                    self.population.append(new_bacterium)

                    # Schedule next division events
                    self.schedule_event(bacterium.time_to_division,
                                       ("division", bacterium_idx))
                    self.schedule_event(new_bacterium.time_to_division,
                                       ("division", len(self.population) - 1))
                else:
                    # Bacterium dies (remove from population)
                    self.population[bacterium_idx] = None

            # Record current statistics
            active_population = [b for b in self.population if b is not None]
            mutant_count = sum(1 for b in active_population if b.mutant_status > 0)

            self.population_history.append(len(active_population))
            self.mutant_history.append(mutant_count)

            # Call the callback with progress updates if provided
            if callback and len(self.population_history) % 100 == 0:
                callback(len(active_population), mutant_count)

        # Apply antibiotic challenge if enabled
        if self.antibiotic_challenge:
            active_population = [b for b in self.population if b is not None]
            surviving_population = [b for b in active_population if b.phenotype == "resistant"]
            self.population = surviving_population

        # Compute final results
        active_population = [b for b in self.population if b is not None]
        mutant_count = sum(1 for b in active_population if b.mutant_status > 0)
        mutant_proportion = mutant_count / len(active_population) if active_population else 0

        return {
            "population_size": len(active_population),
            "mutant_count": mutant_count,
            "mutant_proportion": mutant_proportion,
            "total_divisions": self.total_divisions,
            "total_mutations": self.total_mutations,
            "population_history": self.population_history,
            "mutant_history": self.mutant_history,
            "final_time": self.current_time
        }


class SimulationRunner:
    """Class to manage multiple simulation runs."""

    def __init__(self):
        """Initialize the simulation runner."""
        self.results = []
        self.raw_results = []

    def run_single_simulation(self, params, callback=None):
        """
        Run a single simulation with the given parameters.

        Args:
            params (dict): Simulation parameters
            callback: Optional progress callback

        Returns:
            dict: Simulation results
        """
        sim = LuriaDelbruckSimulation(
            start_pop_size=params["start_pop_size"],
            end_pop_size=params["end_pop_size"],
            division_distribution=params["division_distribution"],
            mutation_rate=params["mutation_rate"],
            lifespan=params["lifespan"],
            division_mean=params["division_mean"],
            division_std=params["division_std"],
            division_lambda=params["division_lambda"],
            antibiotic_challenge=params["antibiotic_challenge"]
        )

        result = sim.run_simulation(callback)
        return result

    def run_sequential_simulations(self, params, num_runs, progress_callback=None):
        """
        Run multiple simulations sequentially.

        Args:
            params (dict): Simulation parameters
            num_runs (int): Number of simulations to run
            progress_callback: Function to call with progress updates

        Returns:
            list: List of simulation results (mutant proportions)
        """
        self.results = []
        self.raw_results = []

        for i in range(num_runs):
            result = self.run_single_simulation(params)
            self.results.append(result["mutant_proportion"])
            self.raw_results.append(result)

            if progress_callback:
                progress_callback(i + 1, num_runs)

        return self.results

    def _parallel_sim_worker(self, params, result_queue, sim_index):
        """Worker function for parallel simulations."""
        try:
            result = self.run_single_simulation(params)
            result_queue.put((sim_index, result))
        except Exception as e:
            result_queue.put((sim_index, f"Error: {str(e)}"))

    def run_parallel_simulations(self, params, num_runs, progress_callback=None):
        """
        Run multiple simulations in parallel.

        Args:
            params (dict): Simulation parameters
            num_runs (int): Number of simulations to run
            progress_callback: Function to call with progress updates

        Returns:
            list: List of simulation results (mutant proportions)
        """
        self.results = [None] * num_runs
        self.raw_results = [None] * num_runs

        # Create a manager for sharing data between processes
        manager = mp.Manager()
        result_queue = manager.Queue()

        # Start worker processes
        processes = []
        for i in range(num_runs):
            p = mp.Process(target=self._parallel_sim_worker,
                          args=(params, result_queue, i))
            processes.append(p)
            p.start()

        # Collect results as they complete
        completed = 0
        while completed < num_runs:
            if not result_queue.empty():
                sim_index, result = result_queue.get()
                if isinstance(result, dict):
                    self.results[sim_index] = result["mutant_proportion"]
                    self.raw_results[sim_index] = result
                    completed += 1

                    if progress_callback:
                        progress_callback(completed, num_runs)
            else:
                time.sleep(0.1)

        # Wait for all processes to finish
        for p in processes:
            p.join()

        return self.results

    def plot_results(self, ax=None):
        """
        Plot the distribution of mutant proportions.

        Args:
            ax: Optional matplotlib axis to plot on

        Returns:
            matplotlib.figure.Figure: The figure containing the plot
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.figure

        if not self.results:
            ax.text(0.5, 0.5, "No simulation results to display",
                   ha="center", va="center", transform=ax.transAxes)
            return fig

        # Plot histogram
        ax.hist(self.results, bins=20, density=True, alpha=0.6, color="skyblue")

        # Add KDE if we have enough data points
        if len(self.results) >= 3:
            try:
                kde = stats.gaussian_kde(self.results)
                x = np.linspace(0, max(self.results) * 1.1, 1000)
                ax.plot(x, kde(x), 'r-', lw=2)
            except Exception:
                # KDE might fail for certain distributions
                pass

        ax.set_xlabel("Proportion of Mutants")
        ax.set_ylabel("Density")
        ax.set_title("Luria-Delbrück Experiment: Distribution of Mutant Proportions")

        return fig


class LuriaDelbruckApp(ctk.CTk):
    """GUI application for the Luria-Delbrück experiment simulation."""

    def __init__(self):
        """Initialize the application."""
        super().__init__()

        self.title("Luria-Delbrück Experiment Simulation")
        self.geometry("1200x800")
        self.minsize(1000, 700)

        # Initialize simulation runner
        self.simulation_runner = SimulationRunner()

        # Create main layout
        self.create_widgets()

        # Thread for running simulations
        self.simulation_thread = None
        self.is_simulation_running = False

    def create_widgets(self):
        """Create and arrange the GUI widgets."""
        # Main layout with two frames
        self.grid_columnconfigure(0, weight=1)
        self.grid_columnconfigure(1, weight=3)
        self.grid_rowconfigure(0, weight=1)

        # Left panel (controls)
        self.control_frame = ctk.CTkFrame(self)
        self.control_frame.grid(row=0, column=0, padx=10, pady=10, sticky="nsew")

        # Right panel (results and visualization)
        self.result_frame = ctk.CTkFrame(self)
        self.result_frame.grid(row=0, column=1, padx=10, pady=10, sticky="nsew")

        # Configure layouts
        self.setup_control_panel()
        self.setup_result_panel()

    def setup_control_panel(self):
        """Set up the control panel with input fields and buttons."""
        # Configure grid
        self.control_frame.grid_columnconfigure(0, weight=1)
        self.control_frame.grid_columnconfigure(1, weight=1)

        # Title
        title_label = ctk.CTkLabel(self.control_frame, text="Simulation Parameters",
                                  font=ctk.CTkFont(size=16, weight="bold"))
        title_label.grid(row=0, column=0, columnspan=2, pady=(10, 20))

        # Population parameters
        row = 1
        ctk.CTkLabel(self.control_frame, text="Initial Population:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.start_pop_var = ctk.StringVar(value="10")
        start_pop_entry = ctk.CTkEntry(self.control_frame, textvariable=self.start_pop_var)
        start_pop_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        row += 1
        ctk.CTkLabel(self.control_frame, text="Target Population:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.end_pop_var = ctk.StringVar(value="1000")
        end_pop_entry = ctk.CTkEntry(self.control_frame, textvariable=self.end_pop_var)
        end_pop_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        # Division parameters
        row += 1
        ctk.CTkLabel(self.control_frame, text="Division Distribution:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.division_dist_var = ctk.StringVar(value="gaussian")
        division_dist_menu = ctk.CTkOptionMenu(self.control_frame, values=["gaussian", "poisson"],
                                             variable=self.division_dist_var,
                                             command=self.on_distribution_change)
        division_dist_menu.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        row += 1
        self.dist_param_frame = ctk.CTkFrame(self.control_frame)
        self.dist_param_frame.grid(row=row, column=0, columnspan=2, padx=5, pady=5, sticky="ew")
        self.setup_distribution_params()

        # Bacterial parameters
        row += 1
        ctk.CTkLabel(self.control_frame, text="Mutation Rate:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.mutation_rate_var = ctk.StringVar(value="1e-6")
        mutation_rate_entry = ctk.CTkEntry(self.control_frame, textvariable=self.mutation_rate_var)
        mutation_rate_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        row += 1
        ctk.CTkLabel(self.control_frame, text="Bacterial Lifespan:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.lifespan_var = ctk.StringVar(value="20")
        lifespan_entry = ctk.CTkEntry(self.control_frame, textvariable=self.lifespan_var)
        lifespan_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        row += 1
        self.antibiotic_var = ctk.BooleanVar(value=False)
        antibiotic_check = ctk.CTkCheckBox(self.control_frame, text="Apply Antibiotic Challenge",
                                         variable=self.antibiotic_var)
        antibiotic_check.grid(row=row, column=0, columnspan=2, padx=5, pady=10, sticky="w")

        # Simulation control
        row += 1
        ctk.CTkLabel(self.control_frame, text="Number of Runs:").grid(row=row, column=0, padx=5, pady=5, sticky="w")
        self.num_runs_var = ctk.StringVar(value="10")
        num_runs_entry = ctk.CTkEntry(self.control_frame, textvariable=self.num_runs_var)
        num_runs_entry.grid(row=row, column=1, padx=5, pady=5, sticky="ew")

        row += 1
        self.parallel_var = ctk.BooleanVar(value=True)
        parallel_check = ctk.CTkCheckBox(self.control_frame, text="Run in Parallel",
                                       variable=self.parallel_var)
        parallel_check.grid(row=row, column=0, columnspan=2, padx=5, pady=10, sticky="w")

        # Run button
        row += 1
        self.run_button = ctk.CTkButton(self.control_frame, text="Run Simulations",
                                      command=self.run_simulations)
        self.run_button.grid(row=row, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

        # Progress bar
        row += 1
        self.progress_bar = ctk.CTkProgressBar(self.control_frame)
        self.progress_bar.grid(row=row, column=0, columnspan=2, padx=20, pady=10, sticky="ew")
        self.progress_bar.set(0)

        # Status label
        row += 1
        self.status_var = tk.StringVar(value="Ready")
        status_label = ctk.CTkLabel(self.control_frame, textvariable=self.status_var)
        status_label.grid(row=row, column=0, columnspan=2, padx=5, pady=5)

        # Save results button
        row += 1
        self.save_button = ctk.CTkButton(self.control_frame, text="Save Results",
                                       command=self.save_results,
                                       state="disabled")
        self.save_button.grid(row=row, column=0, columnspan=2, padx=20, pady=20, sticky="ew")

    def setup_distribution_params(self):
        """Set up the parameters specific to the selected distribution."""
        # Clear existing widgets
        for widget in self.dist_param_frame.winfo_children():
            widget.destroy()

        # Configure grid
        self.dist_param_frame.grid_columnconfigure(0, weight=1)
        self.dist_param_frame.grid_columnconfigure(1, weight=1)

        distribution = self.division_dist_var.get()

        if distribution == "gaussian":
            # Gaussian parameters
            ctk.CTkLabel(self.dist_param_frame, text="Mean:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.division_mean_var = ctk.StringVar(value="10")
            mean_entry = ctk.CTkEntry(self.dist_param_frame, textvariable=self.division_mean_var)
            mean_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

            ctk.CTkLabel(self.dist_param_frame, text="Std Dev:").grid(row=1, column=0, padx=5, pady=5, sticky="w")
            self.division_std_var = ctk.StringVar(value="2")
            std_entry = ctk.CTkEntry(self.dist_param_frame, textvariable=self.division_std_var)
            std_entry.grid(row=1, column=1, padx=5, pady=5, sticky="ew")

        elif distribution == "poisson":
            # Poisson parameter
            ctk.CTkLabel(self.dist_param_frame, text="Lambda:").grid(row=0, column=0, padx=5, pady=5, sticky="w")
            self.division_lambda_var = ctk.StringVar(value="10")
            lambda_entry = ctk.CTkEntry(self.dist_param_frame, textvariable=self.division_lambda_var)
            lambda_entry.grid(row=0, column=1, padx=5, pady=5, sticky="ew")

    def on_distribution_change(self, _):
        """Handle change in the division distribution selection."""
        self.setup_distribution_params()

    def setup_result_panel(self):
        """Set up the results panel with plots and statistics."""
        # Configure grid
        self.result_frame.grid_columnconfigure(0, weight=1)
        self.result_frame.grid_rowconfigure(0, weight=0)  # Header
        self.result_frame.grid_rowconfigure(1, weight=1)  # Plot
        self.result_frame.grid_rowconfigure(2, weight=0)  # Stats

        # Results header
        header_label = ctk.CTkLabel(self.result_frame, text="Simulation Results",
                                   font=ctk.CTkFont(size=16, weight="bold"))
        header_label.grid(row=0, column=0, pady=(10, 5))

        # Plot area
        self.plot_frame = ctk.CTkFrame(self.result_frame)
        self.plot_frame.grid(row=1, column=0, padx=10, pady=5, sticky="nsew")

        # Initialize plot
        self.fig, self.ax = plt.subplots(figsize=(8, 6))
        self.ax.text(0.5, 0.5, "Run a simulation to see results",
                    ha="center", va="center", transform=self.ax.transAxes)

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.canvas.draw()
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(fill=tk.BOTH, expand=True)

        # Statistics panel
        self.stats_frame = ctk.CTkFrame(self.result_frame)
        self.stats_frame.grid(row=2, column=0, padx=10, pady=10, sticky="ew")

        # Stats labels
        ctk.CTkLabel(self.stats_frame, text="Statistics",
                    font=ctk.CTkFont(weight="bold")).grid(row=0, column=0, columnspan=4, pady=5)

        # Row 1
        ctk.CTkLabel(self.stats_frame, text="Total Runs:").grid(row=1, column=0, padx=10, pady=5, sticky="w")
        self.total_runs_var = tk.StringVar(value="0")
        ctk.CTkLabel(self.stats_frame, textvariable=self.total_runs_var).grid(row=1, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(self.stats_frame, text="Avg. Mutant %:").grid(row=1, column=2, padx=10, pady=5, sticky="w")
        self.avg_mutant_var = tk.StringVar(value="0.0")
        ctk.CTkLabel(self.stats_frame, textvariable=self.avg_mutant_var).grid(row=1, column=3, padx=10, pady=5, sticky="w")

        # Row 2
        ctk.CTkLabel(self.stats_frame, text="Median Mutant %:").grid(row=2, column=0, padx=10, pady=5, sticky="w")
        self.median_mutant_var = tk.StringVar(value="0.0")
        ctk.CTkLabel(self.stats_frame, textvariable=self.median_mutant_var).grid(row=2, column=1, padx=10, pady=5, sticky="w")

        ctk.CTkLabel(self.stats_frame, text="Variance:").grid(row=2, column=2, padx=10, pady=5, sticky="w")
        self.variance_var = tk.StringVar(value="0.0")
        ctk.CTkLabel(self.stats_frame, textvariable=self.variance_var).grid(row=2, column=3, padx=10, pady=5, sticky="w")

    def get_simulation_params(self):
        """Retrieve and validate simulation parameters from the GUI."""
        try:
            params = {
                "start_pop_size": int(self.start_pop_var.get()),
                "end_pop_size": int(self.end_pop_var.get()),
                "division_distribution": self.division_dist_var.get(),
                "mutation_rate": float(self.mutation_rate_var.get()),
                "lifespan": int(self.lifespan_var.get()),
                "antibiotic_challenge": self.antibiotic_var.get()
            }

            # Distribution-specific parameters
            if params["division_distribution"] == "gaussian":
                params["division_mean"] = float(self.division_mean_var.get())
                params["division_std"] = float(self.division_std_var.get())
                params["division_lambda"] = 10  # Default value
            else:  # poisson
                params["division_lambda"] = float(self.division_lambda_var.get())
                params["division_mean"] = 10  # Default value
                params["division_std"] = 2  # Default value

            # Validate
            if params["start_pop_size"] <= 0 or params["end_pop_size"] <= 0:
                raise ValueError("Population sizes must be positive")

            if params["mutation_rate"] < 0 or params["mutation_rate"] > 1:
                raise ValueError("Mutation rate must be between")

            if params["lifespan"] <= 0:
                raise ValueError("Bacterial lifespan must be positive")

            return params, int(self.num_runs_var.get())

        except ValueError as e:
            messagebox.showerror("Invalid Parameters", str(e))
            return None, None

    def update_progress(self, current, total):
        """Update the progress bar and status label."""
        progress = current / total if total > 0 else 0
        self.progress_bar.set(progress)
        self.status_var.set(f"Running simulations: {current}/{total}")
        self.update_idletasks()

    def run_simulations(self):
        """Run the simulations with the current parameters."""
        if self.is_simulation_running:
            return

        params, num_runs = self.get_simulation_params()
        if params is None or num_runs is None:
            return

        # Disable controls during simulation
        self.run_button.configure(state="disabled")
        self.save_button.configure(state="disabled")
        self.is_simulation_running = True

        # Reset progress
        self.progress_bar.set(0)
        self.status_var.set("Starting simulations...")

        # Start simulation thread
        def simulation_task():
            try:
                if self.parallel_var.get() and num_runs > 1:
                    results = self.simulation_runner.run_parallel_simulations(
                        params, num_runs, self.update_progress)
                else:
                    results = self.simulation_runner.run_sequential_simulations(
                        params, num_runs, self.update_progress)

                # Update GUI in the main thread
                self.after(0, lambda: self.update_results(results))
            except Exception as e:
                self.after(0, lambda: self.handle_simulation_error(str(e)))
            finally:
                self.is_simulation_running = False
                self.after(0, lambda: self.run_button.configure(state="normal"))

        self.simulation_thread = threading.Thread(target=simulation_task)
        self.simulation_thread.daemon = True
        self.simulation_thread.start()

    def update_results(self, results):
        """Update the GUI with simulation results."""
        if not results:
            self.status_var.set("No results to display")
            return

        # Update plot
        self.ax.clear()
        self.simulation_runner.plot_results(self.ax)
        self.canvas.draw()

        # Update statistics
        self.total_runs_var.set(str(len(results)))

        mean_val = np.mean(results)
        median_val = np.median(results)
        variance_val = np.var(results)

        self.avg_mutant_var.set(f"{mean_val:.4%}")
        self.median_mutant_var.set(f"{median_val:.4%}")
        self.variance_var.set(f"{variance_val:.6f}")

        # Update status
        self.status_var.set(f"Completed {len(results)} simulations")
        self.progress_bar.set(1.0)

        # Enable save button
        self.save_button.configure(state="normal")

    def handle_simulation_error(self, error_msg):
        """Handle errors that occur during simulation."""
        messagebox.showerror("Simulation Error", f"An error occurred: {error_msg}")
        self.status_var.set("Error occurred during simulation")
        self.progress_bar.set(0)

    def save_results(self):
        """Save the simulation results to a CSV file."""
        if not self.simulation_runner.results:
            messagebox.showinfo("No Results", "No simulation results to save")
            return

        try:
            # Create a data frame with results
            data = {
                "Run": list(range(1, len(self.simulation_runner.results) + 1)),
                "Mutant_Proportion": self.simulation_runner.results,
                "Population_Size": [result["population_size"] for result in self.simulation_runner.raw_results],
                "Mutant_Count": [result["mutant_count"] for result in self.simulation_runner.raw_results],
                "Total_Divisions": [result["total_divisions"] for result in self.simulation_runner.raw_results],
                "Total_Mutations": [result["total_mutations"] for result in self.simulation_runner.raw_results],
                "Simulation_Time": [result["final_time"] for result in self.simulation_runner.raw_results]
            }

            df = pd.DataFrame(data)

            # Save to file
            filename = f"luria_delbruck_results_{time.strftime('%Y%m%d_%H%M%S')}.csv"
            df.to_csv(filename, index=False)

            messagebox.showinfo("Save Successful", f"Results saved to {filename}")

        except Exception as e:
            messagebox.showerror("Save Error", f"Failed to save results: {str(e)}")


def main():
    """Main function to run the application."""
    # Enable support for multiprocessing in frozen applications
    if hasattr(mp, 'freeze_support'):
        mp.freeze_support()

    # Create and run the application
    app = LuriaDelbruckApp()
    app.mainloop()


if __name__ == "__main__":
    main()
