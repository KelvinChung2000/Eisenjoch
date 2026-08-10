//! Verify driver-skip asymmetry hypothesis. For each top-fanout net, after
//! placing with HeAP and opt_trans respectively, report:
//!   - fanout, HPWL, driver position, sink centroid
//!   - driver displacement from sink centroid (Manhattan)
//!   - whether the driver cell is a "pure driver" in the solve sense
//!     (i.e. has no sink-role nets in the placer's filter set — every
//!     fan-in is a skipped constant/clock/dead net)
//! If the hypothesis holds, opt_trans should show much larger
//! driver-to-centroid displacement than HeAP on these nets.

use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::netlist::NetId;
use nextpnr::packer;
use nextpnr::placer::heap::{PlacerHeap, PlacerHeapCfg};
use nextpnr::placer::opt_trans::{place_opt_trans, OptTransPlacerCfg};
use nextpnr::placer::Placer;
use std::path::Path;

fn load_fresh(chipdb_path: &str, design_path: &str) -> Context {
    let db = ChipDb::load(Path::new(chipdb_path)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(design_path).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");
    ctx
}

#[derive(Debug, Clone)]
struct NetDiag {
    name: String,
    net_idx: NetId,
    fanout: usize,
    hpwl: i32,
    drv_x: i32,
    drv_y: i32,
    centroid_x: f64,
    centroid_y: f64,
    drv_displace: f64,
    sink_rms_spread: f64,
    drv_is_pure: bool,
}

fn is_dead_net(ctx: &Context, net_idx: NetId) -> bool {
    let net = ctx.net(net_idx);
    if !net.is_alive() || !net.has_driver() || net.num_users() == 0 {
        return true;
    }
    let name = ctx.name_of(net.name_id()).to_lowercase();
    // Match opt_trans default filter (skip_constants=true).
    // Constants are either named gnd/vcc/0/1 or have no user bel positions.
    if name == "$false" || name == "$true" || name == "$undef" || name == "gnd" || name == "vcc" {
        return true;
    }
    false
}

fn diag_net(ctx: &Context, net_idx: NetId) -> Option<NetDiag> {
    let net = ctx.net(net_idx);
    if is_dead_net(ctx, net_idx) {
        return None;
    }
    let drv = net.driver()?;
    if !drv.is_valid() {
        return None;
    }
    let drv_bel = ctx.cell(drv.cell).bel()?;
    let drv_loc = drv_bel.loc();
    let fanout = net.num_users();

    // Collect sink positions.
    let mut sink_xs: Vec<i32> = Vec::with_capacity(fanout);
    let mut sink_ys: Vec<i32> = Vec::with_capacity(fanout);
    for u in net.users() {
        if !u.is_valid() {
            continue;
        }
        let Some(bel) = ctx.cell(u.cell).bel() else {
            continue;
        };
        let loc = bel.loc();
        sink_xs.push(loc.x);
        sink_ys.push(loc.y);
    }
    if sink_xs.is_empty() {
        return None;
    }

    // HPWL over driver + sinks.
    let mut mnx = drv_loc.x;
    let mut mxx = drv_loc.x;
    let mut mny = drv_loc.y;
    let mut mxy = drv_loc.y;
    for (&sx, &sy) in sink_xs.iter().zip(sink_ys.iter()) {
        mnx = mnx.min(sx);
        mxx = mxx.max(sx);
        mny = mny.min(sy);
        mxy = mxy.max(sy);
    }
    let hpwl = (mxx - mnx) + (mxy - mny);

    // Sink centroid.
    let n = sink_xs.len() as f64;
    let cx = sink_xs.iter().map(|&x| x as f64).sum::<f64>() / n;
    let cy = sink_ys.iter().map(|&y| y as f64).sum::<f64>() / n;

    // Driver-to-centroid Manhattan.
    let drv_displace = (drv_loc.x as f64 - cx).abs() + (drv_loc.y as f64 - cy).abs();

    // Sink RMS spread from centroid (tells us how "clumped" the sink cloud is).
    let sum_sq = sink_xs
        .iter()
        .zip(sink_ys.iter())
        .map(|(&x, &y)| (x as f64 - cx).powi(2) + (y as f64 - cy).powi(2))
        .sum::<f64>();
    let sink_rms_spread = (sum_sq / n).sqrt();

    // "Pure driver" heuristic: does the driver cell appear as a user on any
    // other live non-constant net? If not, opt_trans's cost function returns
    // 0 for its position and it can't be pulled by any sink cloud.
    let mut drv_is_pure = true;
    'outer: for other in ctx.design.iter_net_indices() {
        if other == net_idx {
            continue;
        }
        if is_dead_net(ctx, other) {
            continue;
        }
        let other_net = ctx.net(other);
        for u in other_net.users() {
            if u.is_valid() && u.cell == drv.cell {
                drv_is_pure = false;
                break 'outer;
            }
        }
    }

    Some(NetDiag {
        name: ctx.name_of(net.name_id()).to_owned(),
        net_idx,
        fanout,
        hpwl,
        drv_x: drv_loc.x,
        drv_y: drv_loc.y,
        centroid_x: cx,
        centroid_y: cy,
        drv_displace,
        sink_rms_spread,
        drv_is_pure,
    })
}

fn collect_top_fanout(ctx: &Context, top_n: usize) -> Vec<NetDiag> {
    let mut diags: Vec<NetDiag> = ctx
        .design
        .iter_net_indices()
        .filter_map(|net_idx| diag_net(ctx, net_idx))
        .collect();
    diags.sort_by(|a, b| b.fanout.cmp(&a.fanout));
    diags.truncate(top_n);
    diags
}

fn place_heap(ctx: &mut Context) {
    let cfg = PlacerHeapCfg::default();
    PlacerHeap.place(ctx, &cfg).expect("heap place");
}

fn place_otb(ctx: &mut Context) {
    let mut cfg = OptTransPlacerCfg::default();
    cfg.max_outer_iters = 50;
    cfg.num_threads = 8;
    place_opt_trans(ctx, &cfg).expect("opt_trans place");
}

fn main() {
    let chipdb_path = std::env::args().nth(1).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin".into()
    });
    let design_path = std::env::args().nth(2).unwrap_or_else(|| {
        "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json".into()
    });

    println!("========== Placing with HeAP ==========");
    let mut ctx_heap = load_fresh(&chipdb_path, &design_path);
    place_heap(&mut ctx_heap);
    let heap_diags = collect_top_fanout(&ctx_heap, 20);

    println!("\n========== Placing with opt_trans (50 iters, 8 threads, softmin on) ==========");
    let mut ctx_ot = load_fresh(&chipdb_path, &design_path);
    place_otb(&mut ctx_ot);

    // Build a lookup of HeAP's net_idx -> NetDiag, then iterate by the
    // SAME net_idx in opt_trans. This keeps the net identity stable.
    let heap_by_idx: std::collections::HashMap<NetId, NetDiag> =
        heap_diags.iter().map(|d| (d.net_idx, d.clone())).collect();

    println!("\n\n========= PER-NET DIAGNOSTIC (top-20 fanout) =========");
    println!(
        "{:<36}  {:>5}  {:>9}  {:>6}  {:>7}  {:>7}  {:>6}",
        "net_name", "fanout", "placer", "hpwl", "drv_disp", "spread", "pure"
    );
    let mut total_heap_hpwl = 0i64;
    let mut total_ot_hpwl = 0i64;
    let mut total_heap_disp = 0.0f64;
    let mut total_ot_disp = 0.0f64;
    for d_heap in &heap_diags {
        let Some(d_ot) = diag_net(&ctx_ot, d_heap.net_idx) else {
            continue;
        };
        let short_name = if d_heap.name.len() > 36 {
            format!("…{}", &d_heap.name[d_heap.name.len() - 35..])
        } else {
            d_heap.name.clone()
        };
        println!(
            "{:<36}  {:>5}  {:>9}  {:>6}  {:>7.1}  {:>7.1}  {:>6}",
            short_name,
            d_heap.fanout,
            "HeAP",
            d_heap.hpwl,
            d_heap.drv_displace,
            d_heap.sink_rms_spread,
            d_heap.drv_is_pure
        );
        println!(
            "{:<36}  {:>5}  {:>9}  {:>6}  {:>7.1}  {:>7.1}  {:>6}",
            "",
            "",
            "opt_trans",
            d_ot.hpwl,
            d_ot.drv_displace,
            d_ot.sink_rms_spread,
            d_ot.drv_is_pure
        );
        total_heap_hpwl += d_heap.hpwl as i64;
        total_ot_hpwl += d_ot.hpwl as i64;
        total_heap_disp += d_heap.drv_displace;
        total_ot_disp += d_ot.drv_displace;
    }

    println!("\n========= TOTALS OVER TOP-20 FANOUT NETS =========");
    println!(
        "  HeAP   sum(hpwl) = {}    sum(drv_displace) = {:.1}",
        total_heap_hpwl, total_heap_disp
    );
    println!(
        "  opt_T  sum(hpwl) = {}    sum(drv_displace) = {:.1}",
        total_ot_hpwl, total_ot_disp
    );
    if total_heap_hpwl > 0 {
        println!(
            "  ratio  hpwl ot/heap = {:.3}   drv_displace ot/heap = {:.3}",
            total_ot_hpwl as f64 / total_heap_hpwl as f64,
            if total_heap_disp > 0.0 {
                total_ot_disp / total_heap_disp
            } else {
                f64::NAN
            }
        );
    }

    // Also count how many of the top-20 have pure drivers (in either placer's
    // solve filter set — it's a static property of the netlist so it should
    // match between placers, but we print both to double-check).
    let pure_count: usize = heap_diags.iter().filter(|d| d.drv_is_pure).count();
    println!(
        "\n  top-20 nets with pure-driver cells: {}/{}",
        pure_count,
        heap_diags.len()
    );
    if pure_count > 0 {
        println!("  (pure-driver hypothesis predicts opt_trans WORSE on these nets specifically)");
    }
}
