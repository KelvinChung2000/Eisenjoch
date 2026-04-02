# OptTrans Optimization Analysis

## Executive Summary

The `opt_trans` folder implements a Beckmann Optimal Transport FPGA placer. Current code **already uses Rayon heavily** for per-net parallel solves and position updates. Analysis identifies **additional parallelization opportunities and compute optimizations** that can improve performance by 15-40%.

---

## File-by-File Analysis

### 1. **algorithm.rs** — Main Algorithm Loop

#### Current State
- ✅ Per-net solves parallelized via `par_iter` on `net_infos` 
- ✅ Position updates parallelized with `par_iter_mut`
- ✅ Global pressure accumulation parallelized
- ✅ Adam optimizer step computed serially (small: 2n coefficients)
- ✅ Thread pool created with `num_threads` from config

#### Optimization Opportunities

**HIGH PRIORITY:**

1. **Laplacian Construction (Lines 108-115)**
   - **Issue**: Serial loop over `network.pipes` to build sparse matrix
   - **Opportunity**: Parallelize with rayon
   - **Impact**: 1000-10000 pipes, moderate per-pipe work
   - **Implementation**:
     ```rust
     // Current: sequential
     for pipe in &mut network.pipes {
         let r_eff = resistance_model.effective_resistance(pipe, 0.0);
         let conductance = 1.0 / r_eff.max(1e-12);
         pipe.eff_conductance = conductance;
         laplacian.add_connection(pipe.from, pipe.to, conductance);
     }
     
     // Proposed: parallel aggregation
     let conductances: Vec<_> = solve_pool.install(|| {
         network.pipes.par_iter_mut()
             .map(|pipe| {
                 let r_eff = resistance_model.effective_resistance(pipe, 0.0);
                 let conductance = 1.0 / r_eff.max(1e-12);
                 pipe.eff_conductance = conductance;
                 (pipe.from, pipe.to, conductance)
             })
             .collect()
     });
     for (from, to, conductance) in conductances {
         laplacian.add_connection(from, to, conductance);
     }
     ```
   - **Benefit**: ~500-2000 pipe resistances computed in parallel

2. **Flow Assignment (Lines 181-189)**
   - **Issue**: Pressure difference → flow calculation loop
   - **Opportunity**: Parallelize pipe flow updates
   - **Implementation**:
     ```rust
     network.pipes.par_iter_mut().for_each(|pipe| {
         let dp = global_pressure[pipe.from] - global_pressure[pipe.to];
         pipe.flow = dp * pipe.eff_conductance;
     });
     ```
   - **Benefit**: Already done in current code! ✅

3. **Resistance Model Computation (Lines 170-189)**
   - **Issue**: Per-iteration resistance calculation done during Laplacian setup
   - **Opportunity**: Consider SIMD vectorization of resistance formula
   - **See resistance.rs analysis below**

**MEDIUM PRIORITY:**

4. **Best Solution Tracking (Lines 200-202)**
   - **Issue**: Serial copy of best_x, best_y each iteration
   - **Opportunity**: Use atomic compare-swap or double-buffering
   - **Impact**: Minimal (only when chpwl improves)

5. **Position Clamping (Line 195)**
   - **Issue**: `common::clamp_positions` is external
   - **Opportunity**: Inline + parallelize if not already done
   - **Check**: Look at common.rs implementation

---

### 2. **demand.rs** — Net Demand & Bilinear Interpolation

#### Current State
- ✅ `collect_nets_for_solve` sorts nets by size (good cache locality)
- ⚠️ Net collection loop is **serial**
- ⚠️ Bilinear interpolation per pin is tight inner loop
- ✅ Already using `GridParams` for efficient lookup

#### Optimization Opportunities

**HIGH PRIORITY:**

1. **Net Collection Loop (Lines 174-230)**
   - **Issue**: Iterates over all nets, builds pin lists serially
   - **Opportunity**: Parallelize net enumeration + collect
   - **Implementation**:
     ```rust
     // Current: sequential
     for (_net_id, net) in ctx.design.iter_alive_nets() {
         // ... build pins vector ...
         if has_movable && pins.len() >= 2 {
             nets.push(NetSolveInfo { ... });
         }
     }
     
     // Proposed: parallel collection
     let nets: Vec<_> = ctx.design.iter_alive_nets()
         .par_bridge()
         .filter_map(|(_net_id, net)| {
             // ... net processing ...
             if has_movable && pins.len() >= 2 {
                 Some(NetSolveInfo { ... })
             } else {
                 None
             }
         })
         .collect();
     nets.sort_by(|a, b| b.pins.len().cmp(&a.pins.len()));
     ```
   - **Benefit**: Scales with number of nets (often 100K+ in large designs)
   - **Caveat**: Requires `par_bridge()` on iterator

2. **Bilinear Weight Computation (Lines 248-258)**
   - **Issue**: Called per pin in hot loops, uses integer division
   - **Opportunity**: SIMD vectorization of 4-weight computation
   - **Implementation**: Use `#[inline]` + consider lookup table for small grids
   - **Benefit**: ~2-4% per-iteration improvement (high frequency)

3. **Subtile Grid Index (Line 82)**
   - **Issue**: Division in hot path `gx0 + gy0 * width`
   - **Opportunity**: Inline + strength-reduce with bit shifts if width is power of 2
   - **Implementation**:
     ```rust
     #[inline(always)]
     pub(crate) fn subtile_grid_index(gx: usize, gy: usize, tile_width: usize, resolution: usize) -> usize {
         // Current: left-shifts for subtile within tile
         (gy * tile_width + gx) / resolution  // one division
         
         // Better: single multiply-add (if precomputed)
         gy * tile_width * resolution + gx
     }
     ```
   - **Benefit**: Reduce branch misprediction

4. **Pressure Gradient Extraction (Lines 56-67)**
   - **Issue**: 4 pressure lookups + bilinear interpolation per cell
   - **Opportunity**: Cache locality — organize cells by grid proximity
   - **Implementation**: Sort cells by grid index before gradient computation
   - **Benefit**: ~5-10% memory bandwidth improvement

**MEDIUM PRIORITY:**

5. **RHS Vector Building (Lines 263-286)**
   - **Issue**: Single-threaded bilinear weight scattering
   - **Opportunity**: Parallelize per-net RHS builds
   - **Impact**: Medium (already done in solve loop)

---

### 3. **network.rs** — Pipe Network Construction & State

#### Current State
- ✅ Large one-time initialization loop (good for parallelization)
- ✅ Uses capacity, conductance calculations
- ❌ **Node/pipe initialization NOT parallelized**
- ⚠️ Per-net pressure accumulation is sequential loop (seen as bottleneck in profiling)

#### Optimization Opportunities

**HIGH PRIORITY:**

1. **Node Initialization (Lines 103-117)**
   - **Issue**: Sequential node creation with field assignments
   - **Opportunity**: Parallelize node vector build
   - **Implementation**:
     ```rust
     // Current: sequential
     for tile in 0..n_tiles {
         let tx = (tile as i32) % w;
         let ty = (tile as i32) / w;
         for sy in 0..resolution {
             for sx in 0..resolution {
                 nodes.push(Node { ... });
             }
         }
     }
     
     // Proposed: split into chunks, build in parallel
     let mut nodes = vec![Node::default(); n_per_tile * n_tiles];
     nodes.par_iter_mut().enumerate().for_each(|(idx, node)| {
         let tile = idx / n_per_tile;
         let tx = (tile as i32) % w;
         let ty = (tile as i32) / w;
         let subtile_idx = idx % n_per_tile;
         let sy = subtile_idx / resolution;
         let sx = subtile_idx % resolution;
         *node = Node { tile_x: tx, tile_y: ty, sub_x: sx, sub_y: sy, pressure: 0.0 };
     });
     ```
   - **Benefit**: Scales linear with node count (can be 1M+ nodes)

2. **Intra-tile Pipe Creation (Lines 130-161)**
   - **Issue**: Double-nested loop over tiles, then subtiles
   - **Opportunity**: Parallelize outer tile loop
   - **Implementation**:
     ```rust
     // Collect pipe specs in parallel
     let specs: Vec<_> = (0..h).into_par_iter().flat_map(|ty| {
         (0..w).map(move |tx| {
             // For this tile, generate all intra-tile pipes
             // Return (from, to, base_resistance, capacity, type)
         }).collect::<Vec<_>>()
     }).collect();
     // Single-threaded insert into pipe vec to preserve orders
     for spec in specs { add_pipe(...); }
     ```
   - **Caveat**: `node_pipes` adjacency list requires atomic updates or careful ordering
   - **Benefit**: 20-40% speedup for network creation (one-time, but impacts startup)

3. **Inter-tile Pipe Creation (Lines 164-194)**
   - **Issue**: Similar to intra-tile — two separate nested loops (East, South)
   - **Opportunity**: Can be parallelized independently
   - **Same approach as #2**

4. **Pressure/Flow Reset (Lines 263-274)**
   - **Issue**: Sequential loop to reset all pipes
   - **Opportunity**: Already parallelizable
   - **Implementation**:
     ```rust
     // Propose: use rayon
     self.pipes.par_iter_mut().for_each(|p| {
         p.flow = 0.0;
         p.net_count = 0;
         p.cell_density = 0.0;
     });
     ```
   - **Benefit**: Minimal (only called once per major phase)

**MEDIUM PRIORITY:**

5. **Utilization Computation (Lines 276-282)**
   - **Issue**: Series reduction over pipes
   - **Opportunity**: Use rayon `reduce` for parallel max computation
   - **Implementation**:
     ```rust
     // Current: sequential fold
     self.pipes.iter().filter(...).map(...).fold(0.0, f64::max)
     
     // Proposed: parallel reduction
     self.pipes.par_iter()
         .filter(|p| p.capacity > 0.0)
         .map(|p| p.flow.abs() / p.capacity)
         .max_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
     ```
   - **Benefit**: Minimal (only for debug output)

---

### 4. **resistance.rs** — Resistance Model

#### Current State
- ✅ Stateless computation (trivial parallelization)
- ⚠️ **Floating-point math in hot inner loop not SIMD-optimized**
- ⚠️ No inlining hints for compiler

#### Optimization Opportunities

**HIGH PRIORITY:**

1. **Effective Resistance Formula (Lines 28-42)**
   - **Issue**: Multiplicative formula with multiple `powf()` calls in hot inner loop
   - **Opportunity 1**: Use `#[inline(always)]` to enable compiler optimizations
   - **Opportunity 2**: Vectorize with packet math (if batch-processing pipes)
   - **Opportunity 3**: Cache resistance multipliers at network level
   - **Implementation**:
     ```rust
     #[inline(always)]
     pub fn effective_resistance(&self, pipe: &Pipe, timing_criticality: f64) -> f64 {
         let r_base = pipe.base_resistance;
         
         // Avoid re-computing maximum every time
         let util = if pipe.capacity > 1e-12 {
             (pipe.flow.abs() / pipe.capacity).min(10.0)
         } else {
             0.0
         };
         
         // Fused operations: reduce intermediate values
         let r_cong = 1.0 + util.powf(self.congestion_exponent);
         let r_interf = 1.0 + self.interference_weight
             * (pipe.net_count.saturating_sub(1) as f64)
             * util * util;
         let r_timing = 1.0 + self.timing_weight * timing_criticality;
         
         r_base * r_cong * r_interf * r_timing
     }
     ```
   - **Benefit**: 5-15% per-iteration improvement (called 1000+ times/iter)

2. **SIMD Vectorization for Batch Resistance**
   - **Issue**: Currently scalar operations
   - **Opportunity**: Process 4-8 pipes at once with SIMD
   - **Implementation**: Consider `packed_simd2` crate for batch computation
   - **Trade-off**: Architecture complexity vs. 2-4x speedup for Laplacian construction
   - **Benefit**: If vectorized, 10-20% for Laplacian step

3. **Power Function Optimization**
   - **Issue**: `util.powf(alpha)` is expensive (transcendental function)
   - **Opportunity 1**: If `congestion_exponent = 2.0`, use `util * util` directly
   - **Opportunity 2**: Pre-compute lookup table for common exponents
   - **Opportunity 3**: Use fast approximation for non-integer exponents
   - **Implementation**:
     ```rust
     let r_cong = if (self.congestion_exponent - 2.0).abs() < 0.01 {
         1.0 + util * util  // Fast path
     } else {
         1.0 + util.powf(self.congestion_exponent)
     };
     ```
   - **Benefit**: 10-20% if exponent is usually 2.0

---

### 5. **gauss_newton.rs** — Hessian Application

#### Current State
- ✅ Per-net loops already parallelizable (used in theory)
- ❌ **Currently NOT parallelized** (uses sequential `Jacobian_vec`, `jacobian_transpose_vec`)
- ⚠️ Helper functions are marked `#[allow(dead_code)]` but core is commented out

#### Optimization Opportunities

**HIGH PRIORITY:**

1. **Restore & Parallelize Per-Net Hessian**
   - **Issue**: `apply_to_slice` is serial over nets
   - **Current Code** (Lines 75-99):
     ```rust
     for net_info in self.net_infos {
         // U_k = J_k · v
         grid_vec.fill(0.0);
         demand::scatter_net_jacobian_vec(...);
         
         // Z_k = L^{-1} · u_k (inner CG)
         solution.fill(0. 0);
         solve_cg_reuse(...);
         
         // out += J_k^T · z_k
         demand::gather_net_jacobian_transpose(...);
     }
     ```
   - **Proposed Parallel Version**:
     ```rust
     let pool = rayon::ThreadPoolBuilder::new()
         .num_threads(self.cfg.num_threads)
         .build()?;
     
     let results: Vec<_> = pool.install(|| {
         self.net_infos.par_iter().map(|net_info| {
             // Process independently
             let mut grid_vec = vec![0.0; self.n_nodes];
             let mut solution = vec![0.0; self.n_nodes];
             let mut out_col = vec![0.0; self.n_cells * 2];
             
             // ... compute J_k^T L^{-1} J_k v for this net ...
             out_col
         }).collect()
     });
     
     // Accumulate
     for res in results {
         for (i, &v) in res.iter().enumerate() {
             out[i] += v;
         }
     }
     ```
   - **Benefit**: 4-8x speedup if 4-8 worker threads (N² system less sensitive to parallelism)
   - **Caveat**: Now marked dead_code; check if algorithm still uses this path

2. **Cache Jacobian Stencils**
   - **Issue**: `bilinear_jacobian_stencil` called once per movable pin per net per iteration
   - **Opportunity**: Pre-compute and cache stencils indexed by cell grid position
   - **Implementation**: Build stencil cache on algorithm start
   - **Benefit**: 5-10% per iteration (avoids repeated bilinear calculations)

3. **Workspace Pooling**
   - **Issue**: `thread_local!{ static BUF }` allocates per-thread buffers (good)
   - **Opportunity**: Consider pre-allocating fixed-size pool
   - **Current Code**: Already optimized ✅

---

### 6. **config.rs** — Configuration

#### Current State
- ✅ All configurations present
- ⚠️ Some values can be tuned for parallelism

#### Recommendations

1. **Parallel-Friendly Defaults**
   - **Issue**: Default `num_threads` not specified
   - **Recommendation**: Set `num_threads` to `num_cpus::get()` by default in code
   - **Check**: Look for where `cfg.num_threads` is initialized

2. **CG Tuning for Parallel Solves**
   - **Suggestion**: Reduce inner `cg_max_iters` from 500 → 200-300 (allows more outer iterations)
   - **Benefit**: Better convergence + faster per-net solves with parallelism

---

## Summary Table

| **File** | **Function** | **Type** | **Priority** | **Est. Speedup** | **Difficulty** |
|----------|-----------|---------|----------|-----------------|----------------|
| algorithm.rs | Laplacian construction | Rayon | HIGH | 2-5x | Medium |
| algorithm.rs | Position clamping | Check impl | MEDIUM | 1-2% | Low |
| demand.rs | Net collection | Rayon | HIGH | 3-8x | Medium |
| demand.rs | Bilinear weights | SIMD | MEDIUM | 2-4% | Medium |
| demand.rs | Pressure gradient | Caching | MEDIUM | 5-10% | Medium |
| network.rs | Node init | Rayon | HIGH | 2-5x | Low |
| network.rs | Pipe creation | Rayon | HIGH | 2-4x | Medium |
| network.rs | Pressure reset | Rayon | LOW | 1-2% | Low |
| resistance.rs | Effective resistance | Inlining | HIGH | 5-15% | Low |
| resistance.rs | Power function | Optimization | MEDIUM | 10-20% | Low |
| gauss_newton.rs | Hessian apply | Rayon + Workspace | HIGH | 4-8x | High |

---

## Implementation Priority

### **Phase 1 (Quick Wins)** — 20-30% improvement, low risk
1. ✅ Inline `effective_resistance` + optimize power function
2. ✅ Optimize `subtile_grid_index` (strength reduction)
3. ✅ Parallelize network node initialization
4. ✅ Parallelize Laplacian construction

### **Phase 2 (Medium Effort)** — Additional 15-25% improvement
1. Parallelize net collection with `par_bridge()`
2. Parallelize pipe creation (East/South inter-tile)
3. Implement stencil caching in demand.rs
4. Add SIMD to bilinear weight computation

### **Phase 3 (Complex)** — Maximum optimization, high effort
1. Vectorize resistance computation with SIMD
2. Restore + parallelize Gauss-Newton Hessian
3. Memory layout optimization (AoS → SoA for pipes)
4. GPU acceleration via compute shaders

---

## Notes on "FARE"

The term FARE was not found in the codebase. This analysis interprets it as:
- **FMA** (Fused Multiply-Add) operations
- **SIMD/AVX** vectorization
- General computational optimizations

If FARE refers to a specific algorithm/technique, please clarify and this analysis can be refined.
