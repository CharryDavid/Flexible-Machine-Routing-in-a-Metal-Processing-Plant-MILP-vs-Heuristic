# Flexible Machine Routing in a Metal Processing Plant (MILP vs. Heuristic)

This repository contains the code and report for a project on flexible machine routing in a metal processing plant. The project was developed as part of an Industrial Automation course.

## Project Overview

This project addresses a complex scheduling problem in a manufacturing environment. The goal is to schedule a set of jobs that must be processed through three stages on one of five available machines. The routing is flexible, meaning any machine can be used for any stage.

The primary objective is to **minimize the makespan**, which is the total time required to complete all jobs.

Two approaches were developed and compared:
1.  A **Mixed-Integer Linear Programming (MILP)** model to find the mathematically optimal schedule.
2.  A **greedy heuristic** algorithm that provides a fast and near-optimal solution.

The project demonstrates the trade-off between optimality and computational time. While the MILP model guarantees the best solution, it is only feasible for a small number of jobs. The heuristic, on the other hand, can quickly generate good schedules for a larger number of jobs.

## Features

*   **MILP Model:** An exact optimization model formulated using Mixed-Integer Linear Programming.
*   **Heuristic Algorithm:** A fast and efficient greedy heuristic for scheduling.
*   **Gantt Chart Visualization:** The MATLAB script generates a Gantt chart to visualize the resulting schedule.
*   **Detailed Report:** A comprehensive project report in PDF format explains the problem, the models, and the results.

## Requirements

To run the code in this repository, you will need:

*   **MATLAB:** The code is written in the MATLAB programming language.
*   **MATLAB Optimization Toolbox:** The MILP model uses the `intlinprog` function, which is part of the Optimization Toolbox.

## How to Run

1.  Clone this repository to your local machine.
2.  Open MATLAB.
3.  Navigate to the directory where you cloned the repository.
4.  Open the `MILP_BenedictMartus_DavidCharry.m` file in the MATLAB editor.
5.  You can modify the parameters at the beginning of the script, such as the number of jobs (`n_jobs`).
6.  Click the **"Run"** button in the MATLAB editor to execute the script.

The script will print the results of both the MILP and heuristic solutions to the console and generate a Gantt chart for the heuristic schedule.

## Project Structure

*   `MILP_BenedictMartus_DavidCharry.m`: The main MATLAB script containing the implementation of the MILP model and the heuristic algorithm.
*   `Project Report.pdf`: The detailed project report.
*   `README.md`: This file.

## Authors

*   David Charry
*   Benedict Martus

## License
