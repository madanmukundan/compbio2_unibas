import tkinter as tk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import customtkinter as ctk
from queue import PriorityQueue
import time
from collections import deque
import random
from scipy.stats import gaussian_kde

# Set appearance mode and default color theme for customtkinter
ctk.set_appearance_mode("System")
ctk.set_default_color_theme("blue")

class Bacterium:
    """
    Represents a single bacterium in the Luria-Delbrück experiment simulation.
    """
    def __init__(self, phenotype="sensitive", mutation_rate=1e-2, lifespan=20, time_to_division=None, 
                 division_distribution="gaussian", distribution_params=(5, 2), mutant_status=0):
        """
        Initialize a bacterium with specific properties.
        
        Args:
            phenotype (str): Either "sensitive" or "resistant"
            mutation_rate (float): Probability of mutation per division event
            lifespan (int): Number of divisions until death
            time_to_division (int): Number of time steps until next division
            division_distribution (str): Distribution for division time ("gaussian" or "poisson")
            distribution_params (tuple): Parameters for the distribution
            mutant_status (int): 0 if non-mutated, 1 if mutated
        """
        self.age = 0
        self.phenotype = phenotype
        self.mutation_rate = mutation_rate
        self.mutant_status = mutant_status
        self.lifespan = lifespan
        self.division_distribution = division_distribution
        self.distribution_params = distribution_params
        
        # Set initial time to division
        if time_to_division is None:
            self.time_to_division = self._get_division_time()
        else:
            self.time_to_division = time_to_division
    
    def _get_division_time(self):
        """Generate the time to next division based on specified distribution."""
        if self.division_distribution == "gaussian":
            mean, std = self.distribution_params
            # Ensure time is positive by using absolute value or max(1, value)
            return max(1, int(np.random.normal(mean, std)))
        elif self.division_distribution == "poisson":
            lam = self.distribution_params[0]
            return max(1, np.random.poisson(lam))
        else:
            # Default behavior
            return max(1, int(np.random.normal(5, 2)))
    
    def check_mutation(self):
        """Check if bacterium mutates during division."""
        if random.random() < self.mutation_rate:
            if self.phenotype == "sensitive":
                self.phenotype = "resistant"
            else:
                self.phenotype = "sensitive"
            self.mutant_status = 1
            return True
        return False
    
    def divide(self):
        """
        Perform division and return a new daughter bacterium.
        Also resets the parent's time to division.
        """
        # Increment age
        self.age += 1
        
        # Create a daughter bacterium with the same properties
        daughter = Bacterium(
            phenotype=self.phenotype,
            mutation_rate=self.mutation_rate,
            lifespan=self.lifespan,
            time_to_division=self._get_division_time(),
            division_distribution=self.division_distribution,
            distribution_params=self.distribution_params,
            mutant_status=self.mutant_status
        )
        
        # Check for mutation in parent
        self.check_mutation()
        
        # Check for mutation in daughter
        daughter.check_mutation()
        
        # Reset parent's time to division
        self.time_to_division = self._get_division_time()
        
        return daughter
    
    def is_dead(self):
        """Check if bacterium has reached its lifespan."""
        return self.age >= self.lifespan
    
    def decrease_time_to_division(self):
        """Decrease time to division by 1."""
        self.time_to_division -= 1
        return self.time_to_division <= 0

class Event:
    """
    Represents an event in the simulation.
    """
    def __init__(self, time, bacterium_id, event_type):
        self.time = time
        self.bacterium_id = bacterium_id
        self.event_type = event_type  # "division", "death", "check"
    
    def __lt__(self, other):
        """For priority queue ordering."""
        return self.time < other.time

class LuriaDelbruckSimulation:
    """
    Simulation of the Luria-Delbrück experiment.
    """
    def __init__(self, start_pop_size=100, end_pop_size=10000, 
                 division_distribution="gaussian", 
                 distribution_params=(5, 2),
                 mutation_rate=1e-2, 
                 lifespan=20):
        """
        Initialize the simulation with specified parameters.
        
        Args:
            start_pop_size (int): Initial population size
            end_pop_size (int): Target population size to end simulation
            division_distribution (str): Distribution for division times
            distribution_params (tuple): Parameters for the division distribution
            mutation_rate (float): Probability of mutation per division
            lifespan (int): Number of divisions until death
        """
        self.start_pop_size = start_pop_size
        self.end_pop_size = end_pop_size
        self.division_distribution = division_distribution
        self.distribution_params = distribution_params
        self.mutation_rate = mutation_rate
        self.lifespan = lifespan
        
        # Initialize data structures
        self.population = {}  # Dict of bacterium_id: Bacterium
        self.event_queue = PriorityQueue()
        self.current_time = 0
        self.next_id = 0
        self.population_history = []
        self.mutant_counts = []
        
        # For tracking results
        self.final_proportion_mutants = []
    
    def initialize_population(self):
        """Initialize the starting population."""
        self.population = {}
        self.event_queue = PriorityQueue()
        self.current_time = 0
        self.next_id = 0
        self.population_history = []
        self.mutant_counts = []
        
        # Create initial bacteria
        for _ in range(self.start_pop_size):
            bacterium = Bacterium(
                phenotype="sensitive",
                mutation_rate=self.mutation_rate,
                lifespan=self.lifespan,
                division_distribution=self.division_distribution,
                distribution_params=self.distribution_params
            )
            self.population[self.next_id] = bacterium
            
            # Schedule first division check for this bacterium
            self.schedule_event(self.next_id, "check", 1)
            self.next_id += 1
        
        # Record initial state
        self._record_state()
    
    def schedule_event(self, bacterium_id, event_type, time_delta):
        """Schedule an event for a bacterium."""
        event_time = self.current_time + time_delta
        event = Event(event_time, bacterium_id, event_type)
        self.event_queue.put(event)
    
    def _record_state(self):
        """Record the current state of the population."""
        total_count = len(self.population)
        mutant_count = sum(1 for b in self.population.values() if b.mutant_status > 0)
        
        self.population_history.append(total_count)
        self.mutant_counts.append(mutant_count)
    
    def run_simulation(self, callback=None):
        """
        Run the simulation until the population reaches end_pop_size.
        
        Args:
            callback: Optional function to call after each event for UI updates
        
        Returns:
            float: Proportion of mutants in the final population
        """
        self.initialize_population()
        
        while len(self.population) < self.end_pop_size and not self.event_queue.empty():
            event = self.event_queue.get()
            
            # Update current time
            self.current_time = event.time
            
            # Process event
            if event.bacterium_id in self.population:
                bacterium = self.population[event.bacterium_id]
                
                if event.event_type == "division":
                    # Create daughter bacterium
                    daughter = bacterium.divide()
                    self.population[self.next_id] = daughter
                    
                    # Schedule check events
                    self.schedule_event(event.bacterium_id, "check", 1)
                    self.schedule_event(self.next_id, "check", 1)
                    self.next_id += 1
                    
                    # Check if parent should die (reached lifespan)
                    if bacterium.is_dead():
                        del self.population[event.bacterium_id]
                
                elif event.event_type == "check":
                    # Check if it's time to divide
                    if bacterium.decrease_time_to_division():
                        self.schedule_event(event.bacterium_id, "division", 1)
                    else:
                        # Not yet time, schedule another check
                        self.schedule_event(event.bacterium_id, "check", 1)
            
            # Record state periodically (every 100 events)
            if self.event_queue.qsize() % 100 == 0:
                self._record_state()
                
                # Call callback if provided
                if callback:
                    callback(len(self.population), 
                             sum(1 for b in self.population.values() if b.mutant_status > 0))
        
        # Record final state
        self._record_state()
        
        # Calculate proportion of mutants
        total_count = len(self.population)
        mutant_count = sum(1 for b in self.population.values() if b.mutant_status > 0)
        proportion = mutant_count / total_count if total_count > 0 else 0
        
        self.final_proportion_mutants.append(proportion)
        return proportion

class LuriaDelbruckApp(ctk.CTk):
    """
    GUI application for the Luria-Delbrück experiment simulation.
    """
    def __init__(self):
        super().__init__()
        
        # Configure window
        self.title("Luria-Delbrück Experiment Simulation")
        self.geometry("1000x800")
        
        # Instance variables
        self.simulation = None
        self.simulation_results = []
        self.is_running = False
        
        # Create GUI components
        self._create_input_frame()
        self._create_control_frame()
        self._create_results_frame()
        self._create_plot_frame()
    
    def _create_input_frame(self):
        """Create the frame for parameter inputs."""
        input_frame = ctk.CTkFrame(self)
        input_frame.pack(pady=10, padx=10, fill="x")
        
        # Title
        ctk.CTkLabel(input_frame, text="Simulation Parameters", font=("Arial", 16, "bold")).pack(pady=5)
        
        # Parameters frame (2 columns)
        params_frame = ctk.CTkFrame(input_frame)
        params_frame.pack(pady=5, padx=10, fill="x")
        
        # Left column
        left_frame = ctk.CTkFrame(params_frame)
        left_frame.pack(side="left", padx=5, fill="both", expand=True)
        
        # Right column
        right_frame = ctk.CTkFrame(params_frame)
        right_frame.pack(side="right", padx=5, fill="both", expand=True)
        
        # --- Left column fields ---
        # Start population
        start_frame = ctk.CTkFrame(left_frame)
        start_frame.pack(pady=5, fill="x")
        ctk.CTkLabel(start_frame, text="Start Population Size:").pack(side="left", padx=5)
        self.start_pop_entry = ctk.CTkEntry(start_frame)
        self.start_pop_entry.pack(side="right", padx=5)
        self.start_pop_entry.insert(0, "10")
        
        # End population
        end_frame = ctk.CTkFrame(left_frame)
        end_frame.pack(pady=5, fill="x")
        ctk.CTkLabel(end_frame, text="End Population Size:").pack(side="left", padx=5)
        self.end_pop_entry = ctk.CTkEntry(end_frame)
        self.end_pop_entry.pack(side="right", padx=5)
        self.end_pop_entry.insert(0, "1000")
        
        # Mutation rate
        mutation_frame = ctk.CTkFrame(left_frame)
        mutation_frame.pack(pady=5, fill="x")
        ctk.CTkLabel(mutation_frame, text="Mutation Rate:").pack(side="left", padx=5)
        self.mutation_entry = ctk.CTkEntry(mutation_frame)
        self.mutation_entry.pack(side="right", padx=5)
        self.mutation_entry.insert(0, "1e-2")
        
        # --- Right column fields ---
        # Division distribution
        div_dist_frame = ctk.CTkFrame(right_frame)
        div_dist_frame.pack(pady=5, fill="x")
        ctk.CTkLabel(div_dist_frame, text="Division Distribution:").pack(side="left", padx=5)
        self.div_dist_var = tk.StringVar(value="poisson")
        div_dist_dropdown = ctk.CTkOptionMenu(
            div_dist_frame, 
            values=["gaussian", "poisson"],
            variable=self.div_dist_var,
            command=self._update_distribution_params
        )
        div_dist_dropdown.pack(side="right", padx=5)
        
        # Distribution parameters
        self.dist_params_frame = ctk.CTkFrame(right_frame)
        self.dist_params_frame.pack(pady=5, fill="x")
        
        # Lifespan
        lifespan_frame = ctk.CTkFrame(right_frame)
        lifespan_frame.pack(pady=5, fill="x")
        ctk.CTkLabel(lifespan_frame, text="Bacterium Lifespan:").pack(side="left", padx=5)
        self.lifespan_entry = ctk.CTkEntry(lifespan_frame)
        self.lifespan_entry.pack(side="right", padx=5)
        self.lifespan_entry.insert(0, "20")
        
        # Number of simulations
        num_sims_frame = ctk.CTkFrame(input_frame)
        num_sims_frame.pack(pady=5, fill="x")
        ctk.CTkLabel(num_sims_frame, text="Number of Simulations:").pack(side="left", padx=5)
        self.num_sims_entry = ctk.CTkEntry(num_sims_frame)
        self.num_sims_entry.pack(side="right", padx=5)
        self.num_sims_entry.insert(0, "100")
        
        # Initialize the distribution parameters
        self._update_distribution_params("poisson")
    
    def _update_distribution_params(self, distribution):
        """Update the UI for distribution parameters based on selected distribution."""
        # Clear existing widgets
        for widget in self.dist_params_frame.winfo_children():
            widget.destroy()
        
        if distribution == "gaussian":
            # Mean
            ctk.CTkLabel(self.dist_params_frame, text="Mean:").pack(side="left", padx=5)
            self.mean_entry = ctk.CTkEntry(self.dist_params_frame, width=60)
            self.mean_entry.pack(side="left", padx=5)
            self.mean_entry.insert(0, "5")
            
            # Standard deviation
            ctk.CTkLabel(self.dist_params_frame, text="Std Dev:").pack(side="left", padx=5)
            self.std_entry = ctk.CTkEntry(self.dist_params_frame, width=60)
            self.std_entry.pack(side="left", padx=5)
            self.std_entry.insert(0, "2")
            
        elif distribution == "poisson":
            # Lambda
            ctk.CTkLabel(self.dist_params_frame, text="Lambda:").pack(side="left", padx=5)
            self.lambda_entry = ctk.CTkEntry(self.dist_params_frame, width=60)
            self.lambda_entry.pack(side="left", padx=5)
            self.lambda_entry.insert(0, "5")
    
    def _create_control_frame(self):
        """Create the frame for control buttons."""
        control_frame = ctk.CTkFrame(self)
        control_frame.pack(pady=10, padx=10, fill="x")
        
        # Control buttons
        self.run_button = ctk.CTkButton(
            control_frame, 
            text="Run Simulations", 
            command=self._run_simulations
        )
        self.run_button.pack(side="left", padx=10, pady=10, expand=True, fill="x")
        
        self.stop_button = ctk.CTkButton(
            control_frame, 
            text="Stop", 
            command=self._stop_simulations,
            state="disabled"
        )
        self.stop_button.pack(side="left", padx=10, pady=10, expand=True, fill="x")
        
        self.reset_button = ctk.CTkButton(
            control_frame, 
            text="Reset", 
            command=self._reset_simulations
        )
        self.reset_button.pack(side="left", padx=10, pady=10, expand=True, fill="x")
    
    def _create_results_frame(self):
        """Create the frame for displaying simulation progress and results."""
        results_frame = ctk.CTkFrame(self)
        results_frame.pack(pady=10, padx=10, fill="x")
        
        # Progress frame
        progress_frame = ctk.CTkFrame(results_frame)
        progress_frame.pack(pady=5, fill="x")
        
        ctk.CTkLabel(progress_frame, text="Simulation Progress:").pack(side="left", padx=5)
        self.progress_var = tk.DoubleVar(value=0)
        self.progress_bar = ctk.CTkProgressBar(progress_frame)
        self.progress_bar.pack(side="right", padx=5, fill="x", expand=True)
        self.progress_bar.set(0)
        
        # Current simulation stats
        stats_frame = ctk.CTkFrame(results_frame)
        stats_frame.pack(pady=5, fill="x")
        
        ctk.CTkLabel(stats_frame, text="Current Simulation:").grid(row=0, column=0, sticky="w", padx=5)
        self.current_sim_label = ctk.CTkLabel(stats_frame, text="0/0")
        self.current_sim_label.grid(row=0, column=1, sticky="w", padx=5)
        
        ctk.CTkLabel(stats_frame, text="Population:").grid(row=1, column=0, sticky="w", padx=5)
        self.population_label = ctk.CTkLabel(stats_frame, text="0")
        self.population_label.grid(row=1, column=1, sticky="w", padx=5)
        
        ctk.CTkLabel(stats_frame, text="Mutants:").grid(row=1, column=2, sticky="w", padx=5)
        self.mutants_label = ctk.CTkLabel(stats_frame, text="0")
        self.mutants_label.grid(row=1, column=3, sticky="w", padx=5)
        
        ctk.CTkLabel(stats_frame, text="Proportion:").grid(row=1, column=4, sticky="w", padx=5)
        self.proportion_label = ctk.CTkLabel(stats_frame, text="0%")
        self.proportion_label.grid(row=1, column=5, sticky="w", padx=5)
        
        # Results table title
        ctk.CTkLabel(results_frame, text="Simulation Results:", font=("Arial", 14, "bold")).pack(anchor="w", padx=5, pady=5)
        
        # Results table frame
        table_frame = ctk.CTkFrame(results_frame)
        table_frame.pack(pady=5, fill="both", expand=True)
        
        # Create a canvas with scrollbar for results
        self.canvas_results = tk.Canvas(table_frame, borderwidth=0, highlightthickness=0)
        scrollbar = ctk.CTkScrollbar(table_frame, orientation="vertical", command=self.canvas_results.yview)
        self.canvas_results.configure(yscrollcommand=scrollbar.set)
        
        scrollbar.pack(side="right", fill="y")
        self.canvas_results.pack(side="left", fill="both", expand=True)
        
        # Frame inside canvas for results
        self.results_table = ctk.CTkFrame(self.canvas_results)
        self.canvas_results.create_window((0, 0), window=self.results_table, anchor="nw")
        
        # Bind the configure event to update the scrollregion
        self.results_table.bind("<Configure>", self._on_frame_configure)
        
        # Create headers
        ctk.CTkLabel(self.results_table, text="Sim #", width=60).grid(row=0, column=0, padx=5, pady=2, sticky="w")
        ctk.CTkLabel(self.results_table, text="Proportion Mutants", width=160).grid(row=0, column=1, padx=5, pady=2, sticky="w")
    
    def _on_frame_configure(self, event):
        """Update the scroll region when the frame changes size."""
        self.canvas_results.configure(scrollregion=self.canvas_results.bbox("all"))
    
    def _create_plot_frame(self):
        """Create the frame for displaying the plot."""
        self.plot_frame = ctk.CTkFrame(self)
        self.plot_frame.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Create initial empty plot
        self.fig, self.ax = plt.subplots(figsize=(5, 4))
        self.plot_canvas = FigureCanvasTkAgg(self.fig, master=self.plot_frame)
        self.plot_canvas.draw()
        self.plot_canvas.get_tk_widget().pack(fill="both", expand=True)
    
    def _get_distribution_params(self):
        """Get the distribution parameters based on the selected distribution."""
        dist_type = self.div_dist_var.get()
        
        if dist_type == "gaussian":
            return (float(self.mean_entry.get()), float(self.std_entry.get()))
        elif dist_type == "poisson":
            return (float(self.lambda_entry.get()),)
    
    def _update_plot(self, data=None):
        """Update the histogram plot with the simulation results."""
        # Clear previous plot
        self.ax.clear()
        
        if not data or len(data) == 0:
            self.plot_canvas.draw()
            return
        
        # Create the histogram with KDE
        self.ax.hist(data, bins=30, density=True, alpha=0.7, label='Histogram')
        
        # Calculate KDE if we have enough data points
        if len(data) > 10 and [i for i in data if i > 0]:
            kde = gaussian_kde(data)
            x = np.linspace(min(data), max(data), 1000)
            self.ax.plot(x, kde(x), 'r-', label='KDE')
        
        # Customize plot
        self.ax.set_xlabel('Proportion of Mutants')
        self.ax.set_ylabel('Density')
        self.ax.set_title('Distribution of Mutant Proportions')
        self.ax.legend()
        self.ax.grid(True, linestyle='--', alpha=0.7)
        
        # Update the canvas
        self.plot_canvas.draw()
    
    def _update_results_table(self, results):
        """Update the results table with simulation results."""
        # Clear existing results (except headers)
        for widget in self.results_table.winfo_children():
            if widget.grid_info()['row'] != 0:  # Skip headers at row 0
                widget.destroy()
        
        # Add new results
        for i, proportion in enumerate(results):
            ctk.CTkLabel(self.results_table, text=f"{i+1}").grid(row=i+1, column=0, padx=5, pady=2, sticky="w")
            ctk.CTkLabel(self.results_table, text=f"{proportion:.6f}").grid(row=i+1, column=1, padx=5, pady=2, sticky="w")
    
    def _update_simulation_status(self, current_sim, total_sims, population=0, mutants=0):
        """Update the simulation status display."""
        self.current_sim_label.configure(text=f"{current_sim}/{total_sims}")
        self.population_label.configure(text=f"{population}")
        self.mutants_label.configure(text=f"{mutants}")
        
        if population > 0:
            proportion = mutants / population
            self.proportion_label.configure(text=f"{proportion:.6f}")
        else:
            self.proportion_label.configure(text="0")
        
        self.progress_bar.set(current_sim / total_sims if total_sims > 0 else 0)
        
        # Update UI
        self.update()
    
    def _run_simulations(self):
        """Run the specified number of simulations."""
        if self.is_running:
            return
        
        self.is_running = True
        self.run_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        
        # Get simulation parameters
        try:
            start_pop = int(self.start_pop_entry.get())
            end_pop = int(self.end_pop_entry.get())
            mutation_rate = float(self.mutation_entry.get())
            lifespan = int(self.lifespan_entry.get())
            div_dist = self.div_dist_var.get()
            dist_params = self._get_distribution_params()
            num_sims = int(self.num_sims_entry.get())
        except ValueError:
            ctk.CTkMessageBox.show_error("Error", "Invalid parameter values")
            self.is_running = False
            self.run_button.configure(state="normal")
            self.stop_button.configure(state="disabled")
            return
        
        # Reset results
        self.simulation_results = []
        
        # Create a new simulation
        self.simulation = LuriaDelbruckSimulation(
            start_pop_size=start_pop,
            end_pop_size=end_pop,
            division_distribution=div_dist,
            distribution_params=dist_params,
            mutation_rate=mutation_rate,
            lifespan=lifespan
        )
        
        # Schedule simulations
        self.after(100, lambda: self._run_next_simulation(0, num_sims))
    
    def _run_next_simulation(self, current_sim, total_sims):
        """Run the next simulation in the queue."""
        if not self.is_running or current_sim >= total_sims:
            self._finish_simulations()
            return
        
        # Update status
        self._update_simulation_status(current_sim + 1, total_sims)
        
        # Create callback for status updates
        def update_callback(population, mutants):
            self._update_simulation_status(current_sim + 1, total_sims, population, mutants)
        
        # Run simulation
        proportion = self.simulation.run_simulation(update_callback)
        self.simulation_results.append(proportion)
        
        # Update results display
        self._update_results_table(self.simulation_results)
        self._update_plot(self.simulation_results)
        
        # Schedule next simulation
        self.after(100, lambda: self._run_next_simulation(current_sim + 1, total_sims))
    
    def _stop_simulations(self):
        """Stop current simulation."""
        self.is_running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
    
    def _finish_simulations(self):
        """Finish simulation process and update UI."""
        self.is_running = False
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")
        
        # Final update of results
        self._update_results_table(self.simulation_results)
        self._update_plot(self.simulation_results)
    
    def _reset_simulations(self):
        """Reset simulation and clear results."""
        self.is_running = False
        self.simulation_results = []
        
        # Clear UI
        self._update_simulation_status(0, 0)
        self._update_results_table([])
        self._update_plot([])
        
        self.run_button.configure(state="normal")
        self.stop_button.configure(state="disabled")

if __name__ == "__main__":
    app = LuriaDelbruckApp()
    app.mainloop()