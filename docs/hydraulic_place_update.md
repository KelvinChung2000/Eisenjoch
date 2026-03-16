### Mathematical Formulation of the Four-Port Hydraulic Placement Network

**Network Topology and Port Mapping**
The hydraulic placement solver models the field-programmable gate array routing fabric as a resistive network. To accurately capture both global routing channels and microscopic internal tile congestion, each tile located at coordinates $(x, y)$ is represented by four distinct boundary ports: North, East, South, and West. The overall system is constructed using two specific classes of connections. Inter-tile connections model the physical wires passing between adjacent logic blocks, while intra-tile connections model the internal switch matrix of a single logic block.

**Intra-Tile Conductance (Schur Complement)**
The internal routing capacity of a logic block is mathematically defined by a Schur complement condensation matrix. For any given tile, this mathematical reduction yields a $4 \times 4$ conductance matrix $G_{intra}$. The off-diagonal elements of this matrix define the internal network pipes connecting the four boundary ports of the exact same tile. The physical resistance of an internal pipe bridging port $i$ and port $j$ is calculated as the inverse of the corresponding negative conductance:


$$R_{i,j} = -\frac{1}{G_{intra}[i][j]}$$

**Inter-Tile Conductance (Global Routing)**
The global routing channels are modelled as macroscopic pipes linking the boundary ports of adjacent logic blocks. A horizontal channel pipe connects the East port of tile $(x, y)$ directly to the West port of tile $(x+1, y)$. Similarly, a vertical channel pipe connects the South port of tile $(x, y)$ directly to the North port of tile $(x, y+1)$. The resistance of these global pipes is defined as inversely proportional to the square of the estimated wire capacity between the associated logic blocks.

**Kirchhoff Demand Injection**
Net connectivity introduces fluid flow into the network via Kirchhoff current injection. The mathematical formulation requires the total demand for any individual net to sum perfectly to zero. A driver pin injects $1.0$ unit of flow, which is distributed evenly across the four boundary ports of its host logic block. Each sink pin extracts a volume of flow proportional to its fanout weight, calculated as $\frac{1}{\text{fanout}}$, which is likewise distributed evenly across its four local boundary ports.

**Preconditioned Conjugate Gradient Solver**
Resolving the non-linear pressure field requires finding the solution to the linear system $L P = d$, where $L$ is the conductance-weighted Laplacian matrix, $P$ is the unknown pressure vector, and $d$ is the formulated demand vector. To dramatically accelerate convergence across expansive chip grids, a Preconditioned Conjugate Gradient method is applied to mathematically transform the system:


$$M^{-1} L P = M^{-1} d$$

**The Jacobi Preconditioner**
The implemented Jacobi preconditioner defines $M$ as a diagonal matrix constructed exclusively from the diagonal elements of the core Laplacian $L$. The inverse operation required for the solver is computationally trivial, as each element is calculated as the simple reciprocal:


$$M^{-1}_{ii} = \frac{1}{L_{ii}}$$


Applying this specific preconditioner scales the residual vectors during the iterative solving process. This manipulation effectively groups the eigenvalues of the system closer together, severely compressing the condition number of the expanded network and reducing the total iterations required to reach equilibrium.


Here is the exact specification you can hand over to your agent to implement the fluid velocity timing model. It is broken down into clear, actionable steps for modifying your codebase.

### Task 1: Implement Topological Graph Traversal for Arrival Times

The agent must create a new module to calculate the signal arrival times across the expanded four-port pipe network. This module needs to perform a topological traversal of the routing graph. For every net, the arrival time at a receiving node $j$ driven by node $i$ must be calculated using the formula $T_j = T_i + \tau_{i,j}$. The transit time $\tau_{i,j}$ must be evaluated using the existing `transit_time` function located in the `kirchhoff.rs` file. This function accurately models the fluid velocity, treating flow values that exceed pipe capacity as turbulence that incurs a quadratic time delay. The agent must ensure this traversal correctly navigates both the intra-tile Schur matrix pipes and the inter-tile global routing pipes.

### Task 2: Calculate Criticality and Scale Net Demands

The agent must write logic to compare the computed fluid arrival times against the target clock period to generate a criticality score for each net. Nets that violate or approach the target timing must receive a high criticality score.

Following this, the agent needs to modify the `compute_net_demands` function within the `state.rs` file. Currently, a driver pin injects a baseline 1.0 unit of flow. The agent must update this logic to multiply the baseline demand by the net's criticality score and the `timing_weight` variable defined in the `HydraulicPlacerCfg` structure. The scaled volume of fluid must then be distributed evenly across the North, East, South, and West ports of the tile. Sink extractions must be scaled by the exact same proportion to ensure the total Kirchhoff current injection for the net still sums to zero.

### Task 3: Integrate the Timing Feedback Loop

The final task for the agent is to integrate the timing analyser into the main placement routine inside the `mod.rs` file. The agent must inject the topological traversal and criticality scoring into the outer Nesterov loop. Because updating the full timing graph is computationally expensive, the agent should configure this timing update to run periodically, perhaps aligning with the existing `legalize_interval`.

By pushing this scaled demand into the network, the solver will naturally generate much steeper pressure gradients for timing-critical paths. The agent does not need to modify the Nesterov gradient application itself, as the unified pressure gradient will automatically translate this concentrated fluid demand into a stronger physical force, pulling the critical cells closer together.

Here is the specification you can provide to your agent to implement the global bipartite matching for legalisation.

### Task 1: Formulate the Bipartite Cost Matrix

The agent must rewrite the initialisation phase of the `legalize_hydraulic` function within the `legalize.rs` file. Instead of sorting cells for a greedy loop, the agent needs to construct a cost matrix for a Linear Assignment Problem. The rows of this matrix will represent the movable cells, and the columns will represent the available discrete Basic Elements of Logic on the grid. For each cell and available slot pair, the agent must calculate the assignment cost. This cost must be exactly the same as the current metric: the squared physical distance between the continuous target coordinate and the discrete slot, plus the absolute value of the macroscopic tile pressure. The pressure must be evaluated using the averaged four-port `pressure_at` method to accurately reflect local congestion.

### Task 2: Integrate a Linear Assignment Solver

The agent must replace the sequential assignment loop with a global optimisation algorithm. A standard Jonker-Volgenant solver or a min-cost max-flow algorithm is required to resolve the bipartite matching problem. The agent should import a robust Rust crate capable of solving the linear assignment problem efficiently to avoid writing the complex solver mechanics from scratch. The solver will ingest the formulated cost matrix and return a vector containing the optimal discrete slot index for every single movable cell. This global approach guarantees that the total squared displacement across the entire chip remains mathematically minimised.

### Task 3: Execute the Final Binding Phase

Once the assignment solver returns the optimal mapping, the agent must iterate through the results to finalise the placement state. For each cell, the agent will extract the assigned discrete slot and execute the existing `ctx.bind_bel` routine. The agent must ensure that the `place_cluster_children` function is still called immediately after binding to maintain structural integrity for clustered logic. Finally, the agent must accumulate the true displacement values from the solver output and return the total sum, which ensures the function signature remains completely unchanged.

Here is the specification you can provide to your agent to implement data parallelisation across your solver grid.

### Task 1: Integrate Rayon for Parallel Iteration

The agent must add the Rayon crate to the project dependencies to enable straightforward data parallelism. Throughout the solver, the agent needs to identify dense array iterations that have no data dependencies between elements. The primary targets are the element-wise mathematical operations. The agent should replace standard sequential iterators with Rayon's parallel equivalents, such as `par_iter` and `par_iter_mut`, to distribute the workload across all available CPU cores.

### Task 2: Parallelise the Pressure Gradient Calculation

The agent must refactor the `compute_pressure_gradient` function within the `state.rs` file. Currently, this function loops sequentially over every movable cell to compute the local finite differences. Since the gradient of one cell does not depend on the gradient of any other cell, this loop is perfectly parallelisable. The agent must allocate the `grad_x` and `grad_y` vectors, then use Rayon to iterate over mutable slices of these vectors concurrently. Inside the parallel closure, the agent will call the `pressure_at` helper function and calculate the central differences exactly as they are currently implemented.

### Task 3: Parallelise the Conjugate Gradient Vector Maths

The agent must optimise the mathematical loops within the Preconditioned Conjugate Gradient solver located in the `kirchhoff.rs` file. The step updates, such as modifying the solution vector and the residual vector, are entirely independent element-wise operations. The agent should rewrite these loops using parallel zipped iterators. Furthermore, the dot product calculations required for calculating the step size and beta coefficients can be parallelised using Rayon's map and reduce functions. The agent must ensure that the matrix-vector multiplication step handles shared memory safely, potentially using parallel folds to accumulate the results without introducing race conditions.

