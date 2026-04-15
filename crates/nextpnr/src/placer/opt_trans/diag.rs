//! Diagnostic instrumentation for the DCD placer.
//!
//! Enabled by `NPNR_OT_DIAG=1`. Output directory defaults to `/tmp/claude/ot_diag`
//! or `NPNR_OT_DIAG_DIR`. Emits per-sweep CSVs for plateau shape, move magnitude,
//! oscillation history, per-net HPWL, and a final pipe usage histogram.
//!
//! Design goal: zero cost when disabled (single `if !enabled { return; }` at every
//! entry point), and all allocations confined to the sweep boundary.

use std::collections::VecDeque;
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use std::path::PathBuf;

use super::network::PipeNetwork;

const OSC_HISTORY: usize = 3;
const TOP_K: usize = 20;

/// Bounding-box measurements reported per sweep.
#[derive(Clone, Copy, Debug, Default)]
pub struct BoundingBoxes {
    /// BB of movable cells only.
    pub movable_w: i32,
    pub movable_h: i32,
    /// Fill ratio: movable cells / BB area (density of placed logic).
    pub movable_fill: f64,
    /// BB of ALL pin positions (movable + fixed). Use to see IO-driven sprawl.
    pub all_w: i32,
    pub all_h: i32,
}

/// Accumulated plateau statistics from a single fullscan.
#[derive(Clone, Copy, Debug, Default)]
pub struct PlateauStat {
    pub best_cost: f64,
    pub cost_min: f64,
    pub cost_max: f64,
    pub n_within_1pct: u32,
    pub n_within_5pct: u32,
    pub n_tied_at_min: u32,
    pub n_scanned: u32,
}

/// A single committed move (old grid -> new grid position).
#[derive(Clone, Copy, Debug)]
pub struct MoveRecord {
    pub cell_idx: usize,
    pub old_gx: i32,
    pub old_gy: i32,
    pub new_gx: i32,
    pub new_gy: i32,
}

pub struct DiagCtx {
    pub enabled: bool,
    out_dir: PathBuf,
    /// Plateau stats per cell for the current sweep (indexed by cell_idx).
    plateau: Vec<PlateauStat>,
    /// Move records for the current sweep.
    moves: Vec<MoveRecord>,
    /// Per-cell position history, newest-first, for oscillation detection.
    history: Vec<VecDeque<(i32, i32)>>,
    /// Per-net HPWL from previous sweep (for churn diff).
    prev_net_hpwl: Vec<f64>,
    /// Current sweep index.
    sweep_idx: usize,
    /// Cumulative acceptance breakdown.
    total_accept_hpwl_only: usize,
    total_accept_friction_only: usize,
    total_accept_both: usize,
    total_reject: usize,
    /// Summary CSV writer kept across sweeps.
    summary_w: Option<BufWriter<File>>,
}

impl DiagCtx {
    pub fn from_env() -> Self {
        let enabled = std::env::var("NPNR_OT_DIAG").ok().as_deref() == Some("1");
        let out_dir = std::env::var("NPNR_OT_DIAG_DIR")
            .map(PathBuf::from)
            .unwrap_or_else(|_| PathBuf::from("/tmp/claude/ot_diag"));
        if enabled {
            let _ = fs::create_dir_all(&out_dir);
        }
        let summary_w = if enabled {
            let p = out_dir.join("sweep_summary.csv");
            let f = File::create(&p).expect("create summary csv");
            let mut w = BufWriter::new(f);
            writeln!(
                w,
                "sweep,n_cells,n_nets,chpwl,line,friction,max_overflow,n_overflow,overflow_excess,\
                 moves_committed,accept,accept_reason,\
                 d_hpwl,d_friction,\
                 plateau_mean_n_within_1pct,plateau_mean_n_within_5pct,plateau_mean_tied,\
                 plateau_cells_with_ties,plateau_cells_flat,\
                 move_mag_mean,move_mag_p50,move_mag_p90,move_mag_max,\
                 oscillating_cells,\
                 movable_bb_w,movable_bb_h,movable_bb_area,movable_bb_fill,\
                 all_pin_bb_w,all_pin_bb_h"
            )
            .ok();
            Some(w)
        } else {
            None
        };
        Self {
            enabled,
            out_dir,
            plateau: Vec::new(),
            moves: Vec::new(),
            history: Vec::new(),
            prev_net_hpwl: Vec::new(),
            sweep_idx: 0,
            total_accept_hpwl_only: 0,
            total_accept_friction_only: 0,
            total_accept_both: 0,
            total_reject: 0,
            summary_w,
        }
    }

    /// One-shot design-level summary: cell/bucket counts, IO/fixed positions, bounding boxes.
    /// Captures:
    ///  - total design cells by bucket (from ctx.design)
    ///  - movable cell count by bucket (from idx_to_cell / cell_buckets)
    ///  - IO/fixed cell count, positions, bounding box
    ///  - Count of fixed-pin nets (nets with an IO or locked cell on any pin)
    ///  - Chip dimensions and grid dimensions
    pub fn dump_design_summary(&self, lines: &[String]) {
        if !self.enabled {
            return;
        }
        let p = self.out_dir.join("design_summary.txt");
        let Ok(f) = File::create(&p) else { return };
        let mut w = BufWriter::new(f);
        for l in lines {
            let _ = writeln!(w, "{}", l);
        }
    }

    /// One-shot CSV of fixed (IO or locked) cell positions for spatial analysis.
    pub fn dump_fixed_cells(&self, rows: &[(String, String, i32, i32, u32)]) {
        if !self.enabled {
            return;
        }
        let p = self.out_dir.join("fixed_cells.csv");
        let Ok(f) = File::create(&p) else { return };
        let mut w = BufWriter::new(f);
        let _ = writeln!(w, "name,bucket,tile_x,tile_y,fanout");
        for (name, bucket, x, y, fan) in rows {
            let _ = writeln!(w, "{},{},{},{},{}", name, bucket, x, y, fan);
        }
    }

    /// One-shot cell metadata dump.
    /// `rows` is (cell_idx, cell_name, bucket, num_nets, max_fanout_of_any_touching_net,
    ///           total_fanout_sum, has_fixed_pin_on_any_net).
    pub fn dump_cell_metadata(&self, rows: &[(usize, String, String, u32, u32, u32, bool)]) {
        if !self.enabled {
            return;
        }
        let p = self.out_dir.join("cell_metadata.csv");
        let Ok(f) = File::create(&p) else { return };
        let mut w = BufWriter::new(f);
        let _ = writeln!(
            w,
            "cell_idx,cell_name,bucket,num_nets,max_net_fanout,total_fanout,has_fixed_net"
        );
        for (idx, name, bucket, n_nets, max_fan, tot_fan, fixed) in rows {
            let _ = writeln!(
                w,
                "{},{},{},{},{},{},{}",
                idx, name, bucket, n_nets, max_fan, tot_fan, *fixed as i32
            );
        }
    }

    /// Call at the top of each sweep, before Phase 1.
    pub fn sweep_begin(&mut self, n_cells: usize, sweep: usize) {
        if !self.enabled {
            return;
        }
        self.sweep_idx = sweep;
        self.plateau.clear();
        self.plateau.resize(n_cells, PlateauStat::default());
        self.moves.clear();
        if self.history.len() != n_cells {
            self.history = (0..n_cells)
                .map(|_| VecDeque::with_capacity(OSC_HISTORY))
                .collect();
        }
    }

    #[inline]
    pub fn record_plateau(&mut self, cell_idx: usize, stat: PlateauStat) {
        if !self.enabled {
            return;
        }
        if cell_idx < self.plateau.len() {
            self.plateau[cell_idx] = stat;
        }
    }

    #[inline]
    pub fn record_move(&mut self, rec: MoveRecord) {
        if !self.enabled {
            return;
        }
        self.moves.push(rec);
    }

    /// Update per-cell position history with the final (post-Phase-2) grid position.
    pub fn finalize_positions(&mut self, positions: &[(i32, i32)]) {
        if !self.enabled {
            return;
        }
        for (ci, &pos) in positions.iter().enumerate() {
            if ci >= self.history.len() {
                break;
            }
            let h = &mut self.history[ci];
            if h.len() == OSC_HISTORY {
                h.pop_back();
            }
            h.push_front(pos);
        }
    }

    /// Return the number of cells whose current position repeats any position within the last K sweeps.
    fn count_oscillating(&self) -> usize {
        let mut n = 0usize;
        for h in &self.history {
            if h.len() < 2 {
                continue;
            }
            let current = h[0];
            let prior_visit = h.iter().skip(2).any(|&p| p == current);
            if prior_visit {
                n += 1;
            }
        }
        n
    }

    /// Finish a sweep: write per-sweep CSVs and summary row.
    /// `accepted` is the revert decision; `d_hpwl` and `d_friction` are the delta vs best-so-far
    /// (negative = improvement).
    pub fn sweep_end(
        &mut self,
        n_cells: usize,
        chpwl: f64,
        line: f64,
        friction: f64,
        max_overflow: f64,
        n_overflow: usize,
        overflow_excess: f64,
        moves_committed: usize,
        accepted: bool,
        d_hpwl: f64,
        d_friction: f64,
        net_names: &[String],
        net_fanout: &[u32],
        net_hpwl: &[f64],
        bb: &BoundingBoxes,
    ) {
        if !self.enabled {
            return;
        }

        // Accept reason classification.
        let reason_str = if accepted {
            let imp_h = d_hpwl < 0.0;
            let imp_f = d_friction < 0.0;
            match (imp_h, imp_f) {
                (true, true) => {
                    self.total_accept_both += 1;
                    "both"
                }
                (true, false) => {
                    self.total_accept_hpwl_only += 1;
                    "hpwl_only"
                }
                (false, true) => {
                    self.total_accept_friction_only += 1;
                    "friction_only"
                }
                (false, false) => "no_improve",
            }
        } else {
            self.total_reject += 1;
            "reject"
        };

        // Plateau stats.
        let mut sum1 = 0u64;
        let mut sum5 = 0u64;
        let mut sumt = 0u64;
        let mut cells_with_ties = 0usize;
        let mut cells_flat = 0usize; // plateau covers >50% of scanned nodes
        let mut n_scanned = 0u32;
        for s in &self.plateau {
            sum1 += s.n_within_1pct as u64;
            sum5 += s.n_within_5pct as u64;
            sumt += s.n_tied_at_min as u64;
            if s.n_tied_at_min > 1 {
                cells_with_ties += 1;
            }
            if s.n_scanned > 0 && s.n_within_5pct * 2 > s.n_scanned {
                cells_flat += 1;
            }
            n_scanned = n_scanned.max(s.n_scanned);
        }
        let n_valid = self
            .plateau
            .iter()
            .filter(|s| s.n_scanned > 0)
            .count()
            .max(1);
        let mean1 = sum1 as f64 / n_valid as f64;
        let mean5 = sum5 as f64 / n_valid as f64;
        let meant = sumt as f64 / n_valid as f64;

        // Move magnitude stats (Chebyshev distance).
        let mut mags: Vec<u32> = self
            .moves
            .iter()
            .map(|m| (m.new_gx - m.old_gx).abs().max((m.new_gy - m.old_gy).abs()) as u32)
            .collect();
        mags.sort_unstable();
        let move_mag_mean = if mags.is_empty() {
            0.0
        } else {
            mags.iter().map(|&x| x as f64).sum::<f64>() / mags.len() as f64
        };
        let pct = |p: f64| -> u32 {
            if mags.is_empty() {
                0
            } else {
                let idx = ((mags.len() as f64 - 1.0) * p).round() as usize;
                mags[idx.min(mags.len() - 1)]
            }
        };
        let move_mag_p50 = pct(0.50);
        let move_mag_p90 = pct(0.90);
        let move_mag_max = *mags.last().unwrap_or(&0);

        let oscillating = self.count_oscillating();

        if let Some(w) = self.summary_w.as_mut() {
            writeln!(
                w,
                "{},{},{},{:.1},{:.1},{:.1},{:.2},{},{:.2},{},{},{},{:.1},{:.1},\
                 {:.1},{:.1},{:.2},{},{},{:.2},{},{},{},{},\
                 {},{},{},{:.3},{},{}",
                self.sweep_idx,
                n_cells,
                net_names.len(),
                chpwl,
                line,
                friction,
                max_overflow,
                n_overflow,
                overflow_excess,
                moves_committed,
                accepted as i32,
                reason_str,
                d_hpwl,
                d_friction,
                mean1,
                mean5,
                meant,
                cells_with_ties,
                cells_flat,
                move_mag_mean,
                move_mag_p50,
                move_mag_p90,
                move_mag_max,
                oscillating,
                bb.movable_w,
                bb.movable_h,
                bb.movable_w * bb.movable_h,
                bb.movable_fill,
                bb.all_w,
                bb.all_h,
            )
            .ok();
            w.flush().ok();
        }

        // Per-sweep plateau histogram.
        self.write_plateau_csv(n_scanned);

        // Per-sweep move magnitude histogram.
        self.write_move_hist_csv(&mags);

        // Per-sweep top nets by HPWL and by churn.
        self.write_net_top_csv(net_names, net_fanout, net_hpwl);

        // Update prev_net_hpwl for next sweep churn diff.
        self.prev_net_hpwl.clear();
        self.prev_net_hpwl.extend_from_slice(net_hpwl);
    }

    fn write_plateau_csv(&self, n_scanned: u32) {
        if n_scanned == 0 {
            return;
        }
        let p = self
            .out_dir
            .join(format!("plateau_sweep_{:02}.csv", self.sweep_idx));
        let Ok(f) = File::create(&p) else { return };
        let mut w = BufWriter::new(f);
        let _ = writeln!(
            w,
            "cell_idx,best_cost,cost_min,cost_max,n_within_1pct,n_within_5pct,n_tied_at_min,n_scanned"
        );
        for (ci, s) in self.plateau.iter().enumerate() {
            if s.n_scanned == 0 {
                continue;
            }
            let _ = writeln!(
                w,
                "{},{:.3},{:.3},{:.3},{},{},{},{}",
                ci,
                s.best_cost,
                s.cost_min,
                s.cost_max,
                s.n_within_1pct,
                s.n_within_5pct,
                s.n_tied_at_min,
                s.n_scanned
            );
        }
    }

    fn write_move_hist_csv(&self, mags: &[u32]) {
        if mags.is_empty() {
            return;
        }
        let p = self
            .out_dir
            .join(format!("move_hist_sweep_{:02}.csv", self.sweep_idx));
        let Ok(f) = File::create(&p) else { return };
        let mut w = BufWriter::new(f);
        let _ = writeln!(w, "cheb_distance,count");
        let mut prev = mags[0];
        let mut count = 0usize;
        for &m in mags {
            if m != prev {
                let _ = writeln!(w, "{},{}", prev, count);
                prev = m;
                count = 0;
            }
            count += 1;
        }
        let _ = writeln!(w, "{},{}", prev, count);
    }

    fn write_net_top_csv(&self, net_names: &[String], net_fanout: &[u32], net_hpwl: &[f64]) {
        let n = net_hpwl.len();
        if n == 0 {
            return;
        }

        // Top-K by absolute HPWL.
        let mut by_hpwl: Vec<usize> = (0..n).collect();
        by_hpwl.sort_unstable_by(|&a, &b| {
            net_hpwl[b]
                .partial_cmp(&net_hpwl[a])
                .unwrap_or(std::cmp::Ordering::Equal)
        });

        // Top-K by |delta| vs previous sweep.
        let mut by_churn: Vec<(usize, f64)> = if self.prev_net_hpwl.len() == n {
            (0..n)
                .map(|i| (i, (net_hpwl[i] - self.prev_net_hpwl[i]).abs()))
                .collect()
        } else {
            Vec::new()
        };
        by_churn
            .sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let p = self
            .out_dir
            .join(format!("net_top_sweep_{:02}.csv", self.sweep_idx));
        let Ok(f) = File::create(&p) else { return };
        let mut w = BufWriter::new(f);
        let _ = writeln!(w, "rank_kind,rank,net_idx,name,fanout,hpwl,delta_hpwl");
        let k = TOP_K.min(n);
        for r in 0..k {
            let idx = by_hpwl[r];
            let name = net_names.get(idx).map(|s| s.as_str()).unwrap_or("");
            let fanout = net_fanout.get(idx).copied().unwrap_or(0);
            let d = if self.prev_net_hpwl.len() == n {
                net_hpwl[idx] - self.prev_net_hpwl[idx]
            } else {
                0.0
            };
            let _ = writeln!(
                w,
                "hpwl,{},{},{},{},{:.2},{:.2}",
                r, idx, name, fanout, net_hpwl[idx], d
            );
        }
        for r in 0..k.min(by_churn.len()) {
            let (idx, delta) = by_churn[r];
            let name = net_names.get(idx).map(|s| s.as_str()).unwrap_or("");
            let fanout = net_fanout.get(idx).copied().unwrap_or(0);
            let _ = writeln!(
                w,
                "churn,{},{},{},{},{:.2},{:.2}",
                r, idx, name, fanout, net_hpwl[idx], delta
            );
        }
    }

    /// Emit pipe usage histogram and top-K congested pipes at the end of the run.
    pub fn finalize(&mut self, network: &PipeNetwork) {
        if !self.enabled {
            return;
        }

        // Histogram buckets.
        let buckets: [(&str, Box<dyn Fn(f64) -> bool>); 6] = [
            ("0", Box::new(|n| n == 0.0)),
            ("0.5+", Box::new(|n| n >= 0.5)),
            ("1+", Box::new(|n| n >= 1.0)),
            ("2+", Box::new(|n| n >= 2.0)),
            ("5+", Box::new(|n| n >= 5.0)),
            ("10+", Box::new(|n| n >= 10.0)),
        ];
        let p = self.out_dir.join("pipe_usage_hist.csv");
        if let Ok(f) = File::create(&p) {
            let mut w = BufWriter::new(f);
            let _ = writeln!(w, "bucket,count,total_pipes");
            let total = network.pipes.len();
            for (label, pred) in &buckets {
                let count = network.pipes.iter().filter(|p| pred(p.net_count)).count();
                let _ = writeln!(w, "{},{},{}", label, count, total);
            }
        }

        // Top-K congested pipes.
        let mut idx: Vec<usize> = (0..network.pipes.len()).collect();
        idx.sort_unstable_by(|&a, &b| {
            network.pipes[b]
                .net_count
                .total_cmp(&network.pipes[a].net_count)
        });
        let p = self.out_dir.join("pipe_top.csv");
        if let Ok(f) = File::create(&p) {
            let mut w = BufWriter::new(f);
            let _ = writeln!(
                w,
                "rank,pipe_idx,from,to,net_count,capacity,base_resistance,eff_conductance"
            );
            for (r, &i) in idx.iter().take(50).enumerate() {
                let pipe = &network.pipes[i];
                let _ = writeln!(
                    w,
                    "{},{},{},{},{:.3},{:.1},{:.3},{:.3}",
                    r,
                    i,
                    pipe.from,
                    pipe.to,
                    pipe.net_count,
                    pipe.capacity,
                    pipe.base_resistance,
                    pipe.eff_conductance
                );
            }
        }

        // Gini coefficient of pipe usage.
        let mut counts: Vec<f64> = network.pipes.iter().map(|p| p.net_count).collect();
        counts.sort_unstable_by(|a, b| a.total_cmp(b));
        let n = counts.len() as f64;
        let sum: f64 = counts.iter().sum();
        let gini = if sum > 0.0 && n > 0.0 {
            let mut weighted = 0.0f64;
            for (i, &c) in counts.iter().enumerate() {
                let rank = (i + 1) as f64;
                weighted += rank * c;
            }
            (2.0 * weighted) / (n * sum) - (n + 1.0) / n
        } else {
            0.0
        };

        // Acceptance summary.
        let p = self.out_dir.join("acceptance_summary.txt");
        if let Ok(f) = File::create(&p) {
            let mut w = BufWriter::new(f);
            let _ = writeln!(
                w,
                "accept_hpwl_only={}\naccept_friction_only={}\naccept_both={}\nreject={}\npipe_gini={:.4}",
                self.total_accept_hpwl_only,
                self.total_accept_friction_only,
                self.total_accept_both,
                self.total_reject,
                gini
            );
        }

        if let Some(w) = self.summary_w.as_mut() {
            w.flush().ok();
        }
    }
}

/// Helper used inside fullscan: update plateau counters on the fly.
#[inline(always)]
pub fn update_plateau(stat: &mut PlateauStat, cost: f64) {
    if !cost.is_finite() {
        return;
    }
    stat.n_scanned += 1;
    if cost < stat.cost_min {
        stat.cost_min = cost;
    }
    if cost > stat.cost_max {
        stat.cost_max = cost;
    }
}

/// Second pass after the scan: classify each cost against the best to populate the counters.
pub fn classify_plateau(stat: &mut PlateauStat, costs: &[f64], best: f64) {
    stat.best_cost = best;
    let band1 = best.abs() * 0.01;
    let band5 = best.abs() * 0.05;
    let eps = 1e-9_f64.max(best.abs() * 1e-9);
    let mut tied = 0u32;
    let mut w1 = 0u32;
    let mut w5 = 0u32;
    for &c in costs {
        if !c.is_finite() {
            continue;
        }
        let d = c - best;
        if d <= eps {
            tied += 1;
        }
        if d <= band1 {
            w1 += 1;
        }
        if d <= band5 {
            w5 += 1;
        }
    }
    stat.n_tied_at_min = tied;
    stat.n_within_1pct = w1;
    stat.n_within_5pct = w5;
}
