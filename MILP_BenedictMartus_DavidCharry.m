%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Flexible Machine Routing with Re-entrant Verification Stage
% ─────────────────────────────────────────────────────────────────────────
% Five jobs (J1..J5) must pass through three stages.
% Each stage can be processed on any of five machines (M1..M5).
% Transfer times between machines are given by matrix T.
%
% Goal: minimize the total makespan (completion time of last job).
%
% The program compares:
%   ▸ an exact MILP solution for optimal scheduling
%   ▸ a fast heuristic schedule based on earliest start times and machine availability
%
% Random processing times generated uniformly in [5,15].
% Transfer time matrix T defines setup times between machines.
%
% Authors: Benedict Martus, David Charry
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
clear, clc
    %% Parameters
    n_jobs = 6;
    n_machines = 5;
    n_stages = 3;
    p_min = 5;
    p_max = 15;
    % Random seed for reproducibility
    rng(42);
    % Processing times p(j,s,m) randomly in [5,15]
    processing_times = randi([p_min, p_max], n_jobs, n_stages, n_machines);
    % Transfer time matrix T (given)
    T = [0 3 5 4 6;
         3 0 2 4 5;
         5 2 0 3 4;
         4 4 3 0 2;
         6 5 4 2 0];
    %% Calculate Big M dynamically
    max_proc_time_sum = sum(processing_times, 'all');  % sum of all proc times
    max_transfer_time = max(T, [], 'all');              % max transfer time
    BIG_M = max_proc_time_sum + max_transfer_time;     % safe upper bound
    fprintf('Using dynamic BIG_M = %.2f\n', BIG_M);

    %% Variable indexing helper functions
    n_x = n_jobs * n_stages * n_machines; % x_j,s,m
    n_t = n_jobs * n_stages;             % t_j,s

    % --- NEW: Definition and indexing for y variables for general job-stage conflict ---
    % Create a list of all (job, stage) combinations
    job_stage_combinations = zeros(n_jobs * n_stages, 2);
    current_idx = 1;
    for j = 1:n_jobs
        for s = 1:n_stages
            job_stage_combinations(current_idx, :) = [j, s];
            current_idx = current_idx + 1;
        end
    end
    n_js_comb = size(job_stage_combinations, 1); % Total number of unique (j,s) combinations

    % Generate all unique pairs of these (j,s) combinations
    js_pairs_indices = nchoosek(1:n_js_comb, 2); % Indices into job_stage_combinations
    n_js_pairs = size(js_pairs_indices, 1); % Total number of unique pairs of (j,s) combinations

    % y_pair_idx,m: Binary variable for sequencing between two (j,s) combinations on machine m
    n_y = n_js_pairs * n_machines;

    n_vars = n_x + n_t + n_y + 1; % Total number of variables (x, t, y, Cmax)

    % Indexing functions
    idx_x = @(j,s,m) (j-1)*n_stages*n_machines + (s-1)*n_machines + m;
    idx_t = @(j,s) n_x + (j-1)*n_stages + s;
    % New idx_y: mapping from (pair_index, machine) to y variable index
    % pair_idx is the row index in js_pairs_indices
    idx_y = @(pair_idx, m) n_x + n_t + (pair_idx-1)*n_machines + m;
    idx_Cmax = n_vars;

    %% Build constraints
    Aeq = [];
    beq = [];
    A = [];
    b = [];

    % 1) Each job stage assigned exactly one machine
    for j=1:n_jobs
        for s=1:n_stages
            row = zeros(1,n_vars);
            for m=1:n_machines
                row(idx_x(j,s,m)) = 1;
            end
            Aeq = [Aeq; row];
            beq = [beq; 1];
        end
    end

    lb = zeros(n_vars,1);
    ub = inf(n_vars,1);
    ub(1:n_x) = 1; % x variables are binary
    ub(n_x+n_t+1:n_x+n_t+n_y) = 1; % y variables are binary

    % 2) Stage sequencing with transfer times (fixed inequalities)
    for j=1:n_jobs
        for s=1:n_stages-1
            for m1=1:n_machines
                for m2=1:n_machines
                    row = zeros(1,n_vars);
                    % Constraint: t_j,s+1 >= t_j,s + p_j,s,m1 + T(m1,m2)
                    % Activated only if x_j,s,m1 = 1 and x_j,s+1,m2 = 1
                    % Rearranged to A*x <= b form:
                    % -t_j,s+1 + t_j,s <= -p_j,s,m1 - T(m1,m2) + BIG_M*(2 - x_jsm1 - x_j(s+1)m2)
                    row(idx_t(j,s)) = 1;
                    row(idx_t(j,s+1)) = -1;
                    row(idx_x(j,s,m1)) = BIG_M;
                    row(idx_x(j,s+1,m2)) = BIG_M;
                    b = [b; -processing_times(j,s,m1) - T(m1,m2) + 2*BIG_M];
                    A = [A; row];
                end
            end
        end
    end

    % --- NEW: Machine conflict constraints for general job-stage combinations ---
    % A machine can only process one job-stage combination at a time.
    % y_pair_idx,m = 1 if (j1,s1) precedes (j2,s2) on machine m
    % y_pair_idx,m = 0 if (j2,s2) precedes (j1,s1) on machine m

    for p_idx = 1:n_js_pairs
        % Get the original (j,s) values for the current pair
        idx_js1 = js_pairs_indices(p_idx, 1);
        idx_js2 = js_pairs_indices(p_idx, 2);

        j1 = job_stage_combinations(idx_js1, 1);
        s1 = job_stage_combinations(idx_js1, 2);
        j2 = job_stage_combinations(idx_js2, 1);
        s2 = job_stage_combinations(idx_js2, 2);

        for m = 1:n_machines
            % Constraint A: If (j1,s1) precedes (j2,s2) on machine m (y_pair_idx,m = 1)
            % Condition: t_j2,s2 >= t_j1,s1 + p_j1,s1,m
            % Activated if y_pair_idx,m = 1 AND x_j1,s1,m = 1 AND x_j2,s2,m = 1
            % Transformed to A*x <= b:
            % t_j1,s1 - t_j2,s2 <= -p_j1,s1,m + BIG_M * ( (1 - y_pair_idx,m) + (1 - x_j1,s1,m) + (1 - x_j2,s2,m) )
            row_js1_precedes_js2 = zeros(1,n_vars);
            row_js1_precedes_js2(idx_t(j1,s1)) = 1;
            row_js1_precedes_js2(idx_t(j2,s2)) = -1;
            row_js1_precedes_js2(idx_x(j1,s1,m)) = BIG_M;
            row_js1_precedes_js2(idx_x(j2,s2,m)) = BIG_M;
            row_js1_precedes_js2(idx_y(p_idx,m)) = BIG_M; % Coefficient for (1 - y)
            b = [b; -processing_times(j1,s1,m) + 3*BIG_M];
            A = [A; row_js1_precedes_js2];

            % Constraint B: If (j2,s2) precedes (j1,s1) on machine m (y_pair_idx,m = 0)
            % Condition: t_j1,s1 >= t_j2,s2 + p_j2,s2,m
            % Activated if y_pair_idx,m = 0 AND x_j1,s1,m = 1 AND x_j2,s2,m = 1
            % Transformed to A*x <= b:
            % t_j2,s2 - t_j1,s1 <= -p_j2,s2,m + BIG_M * ( y_pair_idx,m + (1 - x_j1,s1,m) + (1 - x_j2,s2,m) )
            row_js2_precedes_js1 = zeros(1,n_vars);
            row_js2_precedes_js1(idx_t(j2,s2)) = 1;
            row_js2_precedes_js1(idx_t(j1,s1)) = -1;
            row_js2_precedes_js1(idx_x(j1,s1,m)) = BIG_M;
            row_js2_precedes_js1(idx_x(j2,s2,m)) = BIG_M;
            row_js2_precedes_js1(idx_y(p_idx,m)) = -BIG_M; % Coefficient for y
            b = [b; -processing_times(j2,s2,m) + 2*BIG_M];
            A = [A; row_js2_precedes_js1];
        end
    end

    %% Correct makespan constraints including processing time
    % Cmax >= t_j,ns + p_j,ns,m_assigned
    for j=1:n_jobs
        for m=1:n_machines
            % Only active if machine m is assigned at last stage for job j
            row = zeros(1,n_vars);
            row(idx_t(j,n_stages)) = 1;              % start time at last stage
            row(idx_x(j,n_stages,m)) = processing_times(j,n_stages,m); % processing time weighted by assignment
            row(idx_Cmax) = -1;
            b = [b; 0];
            A = [A; row];
        end
    end

    % Objective: Minimize Cmax
    f = zeros(n_vars,1);
    f(idx_Cmax) = 1;

    % Integer variables: x and y
    intcon = [1:n_x, (n_x+n_t+1):(n_x+n_t+n_y)];

    % Fixed options line:
    options = optimoptions('intlinprog','Display','iter','Heuristics','advanced',...
                    'CutGeneration','advanced');
    fprintf('Starting MILP solve...\n');
    [z,fval,exitflag,output] = intlinprog(f,intcon,A,b,Aeq,beq,lb,ub,options);

    if exitflag ~= 1
        fprintf('MILP did not find an optimal solution. Exit flag: %d\n', exitflag);
        return;
    end
    fprintf('MILP solved.\n');
    fprintf('Optimal makespan: %.2f\n', fval);

    % Display results
    x_sol = reshape(z(1:n_x), [n_machines, n_stages, n_jobs]);
    x_sol = permute(x_sol, [3 2 1]); % Permute to (job, stage, machine)
    t_sol = reshape(z(n_x+1:n_x+n_t), [n_stages,n_jobs])'; % Permute to (job, stage)

    fprintf('\nDetailed MILP Schedule:\n');
    for j = 1:n_jobs
        fprintf('Job %d:\n', j);
        for s = 1:n_stages
            % Find the assigned machine (x_sol will be 1 for the assigned machine)
            m_assigned = find(round(x_sol(j,s,:)) == 1);
            if isempty(m_assigned)
                fprintf('  Stage %d: No machine assigned (Error in MILP solution)\n', s);
                continue;
            end
            st = t_sol(j,s);
            pt = processing_times(j,s,m_assigned);
            fprintf('  Stage %d: Machine %d, Start: %.2f, Duration: %d, Finish: %.2f\n', ...
                    s, m_assigned, st, pt, st + pt);
        end
    end
    fprintf('Total MILP makespan: %.2f\n', fval);

   %% ---------------- Heuristic Schedule ----------------
fprintf('\nStarting heuristic scheduling...\n');
% Initialization
jobMachineAssignment = zeros(n_jobs, n_stages);  % machine chosen for each job-stage
jobStartTimes = zeros(n_jobs, n_stages);
machineAvailable = zeros(n_machines,1);          % machine next free time

for j = 1:n_jobs
    prev_machine = 0;         % no previous machine for first stage
    prev_finish = 0;          % no previous finish for first stage
    for s = 1:n_stages
        best_start = inf;
        best_machine = 1;
        for m = 1:n_machines
            % earliest start considering:
            % 1) machine availability
            % 2) previous stage finish + transfer time (if not first stage)
            transfer = 0;
            if s > 1 && prev_machine > 0
                transfer = T(prev_machine,m);
            end
            est = max(machineAvailable(m), prev_finish + transfer);
            if est < best_start
                best_start = est;
                best_machine = m;
            end
        end
        % assign chosen machine and start time
        jobMachineAssignment(j,s) = best_machine;
        jobStartTimes(j,s) = best_start;

        % update machine availability
        machineAvailable(best_machine) = best_start + processing_times(j,s,best_machine);
        
        % update prev for next stage (for current job)
        prev_machine = best_machine;
        prev_finish = best_start + processing_times(j,s,best_machine);
    end
end

% Compute makespan
heuristicMakespan = max(max(jobStartTimes + processing_times(sub2ind(size(processing_times), ...
    repmat((1:n_jobs)', 1, n_stages), repmat(1:n_stages, n_jobs, 1), jobMachineAssignment)))); % Completion times of all stages of all jobs
fprintf('Heuristic makespan: %.2f\n', heuristicMakespan);

%% ---------------- Gantt Chart Plot ----------------
figure('Name','Heuristic Schedule Gantt Chart');
hold on;
colors = lines(n_jobs);
yticksLabels = cell(n_jobs*n_stages,1);
yticksPos = zeros(n_jobs*n_stages,1);
barHeight = 0.8;

for j = 1:n_jobs
    for s = 1:n_stages
        m = jobMachineAssignment(j,s);
        startTime = jobStartTimes(j,s);
        dur = processing_times(j,s,m);
        
        % Calculate y position for the bar (offset by job for distinct stages)
        % For plotting clarity, let's group by machine, but the original
        % code grouped by job. Sticking to original grouping for now.
        yPos = (j-1)*n_stages + s; 
        
        yticksLabels{yPos} = sprintf('Job%d Stage%d (M%d)', j, s, m);
        yticksPos(yPos) = yPos;
        
        % Draw bar (horizontal)
        barh(yPos, dur, barHeight, 'FaceColor', colors(j,:), 'EdgeColor', 'k', 'BaseValue', startTime);
    end
end

yticks(yticksPos);
yticklabels(yticksLabels);
xlabel('Time');
title('Heuristic Scheduling Gantt Chart');
grid on;
hold off;

%% ---------------- Report ----------------
fprintf('\nDetailed Heuristic Schedule:\n');
for j=1:n_jobs
    fprintf('Job %d:\n', j);
    for s=1:n_stages
        m = jobMachineAssignment(j,s);
        st = jobStartTimes(j,s);
        pt = processing_times(j,s,m);
        fprintf('  Stage %d: Machine %d, Start: %.2f, Duration: %d, Finish: %.2f\n', ...
                s, m, st, pt, st+pt);
    end
end
fprintf('Total heuristic makespan: %.2f\n', heuristicMakespan);