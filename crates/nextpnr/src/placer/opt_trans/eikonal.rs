//! Fast Sweeping Method (FSM) — Eikonal equation solver on the pipe network.
//!
//! Solves |∇T| = f(x) where f is the local slowness (1/eff_conductance).
//! The 2D Eikonal update allows diagonal wavefront propagation, giving
//! smoother distance fields than graph-based Dijkstra.
//!
//! Grid (span-1) pipes: solved via the continuous Eikonal quadratic update.
//! Long-range pipes: handled as graph relaxation (discrete shortcuts).
//! Early termination: stops once all marked sinks are reached with stable T.

use super::network::{PipeNetwork, PipeType, Direction};
use super::path_solver::PathSolverWorkspace;

/// Maximum full sweep rounds (each = 4 directional sweeps).
const MAX_SWEEP_ROUNDS: usize = 6;

/// Solve the 2D Eikonal update for anisotropic edge costs.
///
/// Given best x-neighbor value `tx` with edge cost `fx`, and best y-neighbor
/// value `ty` with edge cost `fy`, returns the 2D travel time T.
///
/// Solves: ((T - tx)/fx)^2 + ((T - ty)/fy)^2 = 1
///
/// Falls back to 1D solution when the 2D discriminant is negative.
#[inline(always)]
fn eikonal_2d(tx: f64, fx: f64, ty: f64, fy: f64) -> f64 {
    // 1D candidates.
    let t1d = (tx + fx).min(ty + fy);

    // 2D quadratic: A*T^2 - 2*B*T + C = 0
    let inv_fx2 = 1.0 / (fx * fx);
    let inv_fy2 = 1.0 / (fy * fy);
    let a = inv_fx2 + inv_fy2;
    let b = tx * inv_fx2 + ty * inv_fy2;
    let c = tx * tx * inv_fx2 + ty * ty * inv_fy2 - 1.0;

    let disc = b * b - a * c;
    if disc < 0.0 {
        return t1d;
    }

    let t2d = (b + disc.sqrt()) / a;

    // The 2D solution is only valid if T > max(tx, ty) — the wavefront must
    // have passed both neighbors before reaching this node diagonally.
    if t2d > tx && t2d > ty {
        t2d.min(t1d) // take the better of 2D and 1D
    } else {
        t1d
    }
}

/// Run the Fast Sweeping Method from a single source node.
///
/// Produces `ws.dist`, `ws.phys_dist`, `ws.prev_pipe`, and `ws.touched`
/// compatible with `dijkstra_early_terminate`.
///
/// Returns the total number of node updates performed.
pub(crate) fn fsm_solve(
    network: &PipeNetwork,
    source_node: usize,
    n_sinks: usize,
    ws: &mut PathSolverWorkspace,
) -> usize {
    let w = network.width as usize;
    let h = network.height as usize;
    let n_nodes = network.num_nodes();
    let pipes = &network.pipes;
    let node_pipes = &network.node_pipes;

    let dist = &mut ws.dist;
    let phys_dist = &mut ws.phys_dist;
    let prev_pipe = &mut ws.prev_pipe;
    let is_sink = &ws.is_sink;
    let touched = &mut ws.touched;

    // Seed source.
    dist[source_node] = 0.0;
    phys_dist[source_node] = 0.0;
    touched.push(source_node);

    let mut total_updates = 0usize;

    for _round in 0..MAX_SWEEP_ROUNDS {
        let mut changed = false;

        for sweep in 0..4u8 {
            let y_fwd = sweep < 2;
            let x_fwd = sweep % 2 == 0;

            let y_iter: Vec<usize> = if y_fwd {
                (0..h).collect()
            } else {
                (0..h).rev().collect()
            };

            for &iy in &y_iter {
                let x_iter: Vec<usize> = if x_fwd {
                    (0..w).collect()
                } else {
                    (0..w).rev().collect()
                };

                for &ix in &x_iter {
                    let node = iy * w + ix;
                    if node >= n_nodes {
                        continue;
                    }

                    // Collect best x-neighbor and y-neighbor for Eikonal update,
                    // and any long-range pipe candidates.
                    let mut best_tx = f64::INFINITY;
                    let mut best_fx = f64::INFINITY;
                    let mut best_x_pipe = usize::MAX;
                    let mut best_x_neighbor = usize::MAX;

                    let mut best_ty = f64::INFINITY;
                    let mut best_fy = f64::INFINITY;
                    let mut best_y_pipe = usize::MAX;
                    let mut best_y_neighbor = usize::MAX;

                    // Best long-range 1D candidate.
                    let mut best_lr = f64::INFINITY;
                    let mut best_lr_pipe = usize::MAX;
                    let mut best_lr_neighbor = usize::MAX;
                    let mut best_lr_span = 0.0f64;

                    for &pipe_idx in &node_pipes[node] {
                        let pipe = &pipes[pipe_idx];
                        let neighbor = if pipe.from == node {
                            pipe.to
                        } else if pipe.to == node {
                            pipe.from
                        } else {
                            continue;
                        };

                        let g = pipe.eff_conductance;
                        if !g.is_finite() || g <= 0.0 {
                            continue;
                        }
                        let edge_cost = 1.0 / g;

                        if dist[neighbor] == f64::INFINITY {
                            continue; // Neighbor not yet reached.
                        }

                        match pipe.pipe_type {
                            PipeType::InterTile(Direction::East) => {
                                // x-direction pipe.
                                let candidate_t = dist[neighbor];
                                if candidate_t < best_tx {
                                    best_tx = candidate_t;
                                    best_fx = edge_cost;
                                    best_x_pipe = pipe_idx;
                                    best_x_neighbor = neighbor;
                                }
                            }
                            PipeType::InterTile(Direction::South) => {
                                // y-direction pipe.
                                let candidate_t = dist[neighbor];
                                if candidate_t < best_ty {
                                    best_ty = candidate_t;
                                    best_fy = edge_cost;
                                    best_y_pipe = pipe_idx;
                                    best_y_neighbor = neighbor;
                                }
                            }
                            PipeType::LongRange { dx, dy } => {
                                // Graph relaxation for long-range shortcuts.
                                let candidate = dist[neighbor] + edge_cost;
                                if candidate < best_lr {
                                    best_lr = candidate;
                                    best_lr_pipe = pipe_idx;
                                    best_lr_neighbor = neighbor;
                                    best_lr_span = (dx.abs() + dy.abs()) as f64;
                                }
                            }
                        }
                    }

                    // Compute Eikonal update from grid neighbors.
                    let mut new_t = dist[node];
                    let mut new_pipe = prev_pipe[node];
                    let mut new_phys = phys_dist[node];

                    let has_x = best_tx.is_finite() && best_fx.is_finite();
                    let has_y = best_ty.is_finite() && best_fy.is_finite();

                    if has_x && has_y {
                        // 2D Eikonal update.
                        let t_eik = eikonal_2d(best_tx, best_fx, best_ty, best_fy);
                        if t_eik < new_t {
                            new_t = t_eik;
                            // Attribute to the neighbor that contributed more.
                            if best_tx + best_fx <= best_ty + best_fy {
                                new_pipe = best_x_pipe;
                                new_phys = phys_dist[best_x_neighbor] + 1.0;
                            } else {
                                new_pipe = best_y_pipe;
                                new_phys = phys_dist[best_y_neighbor] + 1.0;
                            }
                        }
                    } else if has_x {
                        let t1d = best_tx + best_fx;
                        if t1d < new_t {
                            new_t = t1d;
                            new_pipe = best_x_pipe;
                            new_phys = phys_dist[best_x_neighbor] + 1.0;
                        }
                    } else if has_y {
                        let t1d = best_ty + best_fy;
                        if t1d < new_t {
                            new_t = t1d;
                            new_pipe = best_y_pipe;
                            new_phys = phys_dist[best_y_neighbor] + 1.0;
                        }
                    }

                    // Long-range relaxation.
                    if best_lr < new_t {
                        new_t = best_lr;
                        new_pipe = best_lr_pipe;
                        new_phys = phys_dist[best_lr_neighbor] + best_lr_span;
                    }

                    // Commit if improved.
                    if new_t < dist[node] {
                        if dist[node] == f64::INFINITY {
                            touched.push(node);
                        }
                        dist[node] = new_t;
                        phys_dist[node] = new_phys;
                        prev_pipe[node] = new_pipe;
                        changed = true;
                        total_updates += 1;
                    }
                }
            }
        }

        if !changed {
            break;
        }

        // Early termination: check if all sinks are reached with stable dists.
        if n_sinks > 0 {
            let all_reached = is_sink.iter().enumerate()
                .filter(|(_, &s)| s)
                .all(|(n, _)| dist[n].is_finite());
            if all_reached && !changed {
                break;
            }
        }
    }

    total_updates
}
