//! Single opt_trans run, fully configured from `NPNR_OT_*` env vars, reporting
//! the metrics that decide whether the spreading field earns its place.
//!
//! The headline number is `interior_over_cap`: tile edges whose routing demand
//! exceeds chipdb capacity, counted away from the chip boundary. That is the
//! quantity the FPGA01 investigation could never move — 19,970 at the baseline,
//! 20,500 at BPR alpha=0.05 and 20,830 at alpha=10, i.e. a 200x change in
//! penalty weight moved it by 1.6% in the WRONG direction. If a global
//! spreading potential is the missing mechanism rather than a bigger penalty,
//! this is where it has to show up.
//!
//! Usage:
//!   probe_spread_ab [chipdb] [design]
//! Everything else comes from the environment, so an A/B is a shell loop and
//! the binary never needs rebuilding between arms.

use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::metrics::congestion::estimate_congestion;
use nextpnr::packer;
use nextpnr::placer::opt_trans::{place_opt_trans, OptTransPlacerCfg};
use nextpnr::router::raster::{RasterRouter, RasterRouterCfg};
use nextpnr::router::Router;
use std::path::Path;
use std::time::Instant;

fn main() {
    let chipdb_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let design_path = std::env::args().nth(2).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json".into()
    });
    let arm = std::env::var("ARM").unwrap_or_else(|_| "unnamed".into());
    let do_route = std::env::var("PROBE_ROUTE").ok().as_deref() != Some("0");

    let db = ChipDb::load(Path::new(&chipdb_path)).expect("load chipdb");
    let mut ctx = Context::new(db);

    // `import_ispd_raw.py` preserves the ISPD netlist's own cell types, which
    // do not all name a BEL type in the xc_ultrascale chipdb: that device
    // declares one `LUT6` BEL per LUT slot and one `IOB` per IO slot. The
    // arity-N LUTs are drop-in on a LUT6 slot (gen_ultrascale.py registers
    // LUT1..LUT6 timing variants for exactly this reason) and the buffer types
    // all occupy IO slots.
    //
    // Opt-in rather than automatic: the xc7 designs use the lossy converter's
    // DFF/LUT6 naming and already match their device, so applying this table
    // there would silently retarget cells onto BEL types xc7 does not have.
    if std::env::var("PROBE_ALIASES").as_deref() == Ok("ultrascale") {
        for lut in ["LUT1", "LUT2", "LUT3", "LUT4", "LUT5"] {
            ctx.add_cell_type_alias(lut, "LUT6");
        }
        for io in ["IBUF", "OBUF", "BUFGCE"] {
            ctx.add_cell_type_alias(io, "IOB");
        }
        ctx.add_cell_type_alias("DFF", "FDRE");
        ctx.add_cell_type_alias("DLATCH", "FDRE");
    }

    let json = std::fs::read_to_string(&design_path).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");
    println!(
        "[{}] packed: {} cells, {} nets",
        arm,
        ctx.design.num_cells(),
        ctx.design.num_nets()
    );

    let mut cfg = OptTransPlacerCfg::default();
    cfg.max_outer_iters = std::env::var("PROBE_ITERS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(50);
    // Each pooled solver workspace holds a per-pipe f64 array (2.5M pipes ~=
    // 20MB on xc7_large), so thread count is a direct memory knob, not just a
    // speed one. Lower it when the run has to fit under a cap.
    cfg.num_threads = std::env::var("PROBE_THREADS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(8);
    // apply_env_overrides runs inside place_opt_trans, so NPNR_OT_* wins.

    let t = Instant::now();
    place_opt_trans(&mut ctx, &cfg).expect("place_opt_trans");
    let place_secs = t.elapsed().as_secs_f64();
    let hpwl = nextpnr::metrics::total_hpwl(&ctx);

    // Cell occupancy: the spreading field should show up here first.
    let mut by_tile: std::collections::HashMap<(i32, i32), usize> =
        std::collections::HashMap::new();
    for (_cid, cell) in ctx.design.iter_alive_cells() {
        if let Some(bel) = cell.bel {
            let loc = ctx.bel(bel).loc();
            *by_tile.entry((loc.x, loc.y)).or_insert(0) += 1;
        }
    }
    let occupied_tiles = by_tile.len();
    let max_per_tile = by_tile.values().copied().max().unwrap_or(0);

    // Router-agnostic global routing congestion. Same metric, and the same
    // interior-only accounting, as the FPGA01 measurements this is compared
    // against: the corner spike near CLK_BUFG is a boundary artifact, so
    // boundary edges are excluded rather than allowed to dominate.
    let report = estimate_congestion(&ctx, 1.0);
    let h = &report.h_congestion;
    let v = &report.v_congestion;
    let height = h.len();
    let width = if height > 0 { h[0].len() } else { 0 };
    let mut interior_over = 0usize;
    let mut boundary_over = 0usize;
    let mut sum_ratio = 0.0f64;
    let mut n_ratio = 0usize;
    for y in 0..height {
        for x in 0..width {
            for grid in [h, v] {
                let r = grid[y][x];
                if r <= 0.0 {
                    continue;
                }
                sum_ratio += r;
                n_ratio += 1;
                if r > 1.0 {
                    let is_boundary = x == 0 || y == 0 || x + 1 >= width || y + 1 >= height;
                    if is_boundary {
                        boundary_over += 1;
                    } else {
                        interior_over += 1;
                    }
                }
            }
        }
    }
    let avg_ratio = if n_ratio > 0 {
        sum_ratio / n_ratio as f64
    } else {
        0.0
    };

    println!(
        "[{arm}] RESULT place_s={place_secs:.1} hpwl={hpwl:.0} occupied_tiles={occupied_tiles} \
         max_per_tile={max_per_tile} interior_over_cap={interior_over} \
         boundary_over_cap={boundary_over} avg_demand_over_cap={avg_ratio:.3} \
         max_cong={:.2}",
        report.max_congestion
    );

    if !do_route {
        return;
    }

    let mut rcfg = RasterRouterCfg::default();
    rcfg.max_iterations = 5;
    rcfg.verbose = false;
    let t = Instant::now();
    let res = RasterRouter.route(&mut ctx, &rcfg);
    let route_secs = t.elapsed().as_secs_f64();

    let mut alive = 0usize;
    let mut fully = 0usize;
    let mut total_wl: u64 = 0;
    for net_idx in ctx.design.iter_net_indices() {
        let net = ctx.net(net_idx);
        if !net.is_alive() || !net.has_driver() || net.num_users() == 0 {
            continue;
        }
        alive += 1;
        if net.wires().is_empty() {
            continue;
        }
        let num_pips = net.wires().values().filter(|pm| pm.pip.is_some()).count();
        let mut touched = 0;
        for user in net.users() {
            if !user.is_valid() {
                continue;
            }
            let Some(ubel) = ctx.cell(user.cell).bel() else {
                continue;
            };
            let Some(uw) = ubel.pin_wire(user.port) else {
                continue;
            };
            let uwid = uw.id();
            if net.wires().contains_key(&uwid) {
                touched += 1;
                continue;
            }
            let mut hit = false;
            ctx.chipdb().node_wires_cb(uwid, |nw| {
                if !hit && net.wires().contains_key(&nw) {
                    hit = true;
                }
            });
            if hit {
                touched += 1;
            }
        }
        if touched == net.num_users() {
            fully += 1;
            total_wl += num_pips as u64;
        }
    }

    println!(
        "[{arm}] ROUTE ok={} route_s={route_secs:.1} fully={fully}/{alive} routed_wl={total_wl}",
        res.is_ok()
    );
}
