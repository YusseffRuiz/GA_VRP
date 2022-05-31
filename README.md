# VRP Solutions

On the main File we will find two options, if we choose option 0, the VRP will be solved via Genetic Algorithms, option 1 = Ant Colony Optimization.
Both algorithms based on PariseC code

## Requirements
In the Excel File:
 - id: the number of location to be visited by the store
 - id = 0 : The depot location
 - x and y coordinates: coordinates of each location, will be used to calculate distance
 - demand: how much weight the trucks are going to carry in the specific destination

The values can be change, but we need to make sure is something viable.
If we increase the load, then we need to modify hyperparameter v_cap

To run the optimization problem:
Run "main.py"
Install:
- numpy
- pandas
- xlsxwriter
- matplotlib
- openpyxl

# GA

# Parameters
 - :param filepath:Xlsx file path
 - :param outputFile: Xlsx output file path for results, giving optimized values and the order for the destinations
 - :param epochs:Number of Iterations
 - :param pc:Crossover probability
 - :param pm:Mutation probability
 - :param popsize:Population size
 - :param n_select:Number of excellent individuals selected
 - :param v_cap:Vehicle capacity, how much every vechicle can carry
 - :param opt_type:Optimization type:
      - 0:Minimize the number of vehicles,1:Minimize travel distance
 

# Differences
The differentes done in this repository are commented blocks about what is done in each place.
A little bit of optimization in some functions:
  - calFit: Modified to store the values of vehicles and distance traveled, independentely of what we are looking to minimize
  - InitialSol: Modified to make it simplier.
The distribution was made in different classes for optimization and simplier use in main method.
Results are also written in a separate file and location can be selected from main



## ACO
parameters filepath:Xlsx file path
- :param Q:Total pheromone or TAU
- :param alpha:Information heuristic factor
- :param beta:Expected heuristic factor
- :param rho:Information volatilization factor, decay value between 0 to 1
- :param epochs:Iterations
- :param v_cap: Vehicle capacity
- :param opt_type:Optimization type:0:Minimize the number of vehicles,1:Minimize travel distance
- :param popsize:Population size

