//! Synthetic pile + legalize on FPGA01.
//!
//! Bypasses DCD entirely: loads FPGA01, packs, sets every movable cell's
//! position to the chip centroid, then runs SnapLegalizer. The pile shape
//! is exactly what DCD produces at convergence (centroid init + net pull
//! collapses to one tile), so we get the same legalize input without the
//! 30+ minute DCD compute.
//!
//! Useful for tuning the spread_overcrowded behavior in isolation.

use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use nextpnr::placer::common::TypeAwarePlacement;
use nextpnr::netlist::CellId;
use nextpnr::placer::legalize::SnapLegalizer;
use nextpnr::placer::legalize::Legalizer;
use std::path::Path;

fn main() {
    let chipdb = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin";
    let design = "/home/kelvin/side-project/eisenjoch/benchmark/ispd/generated/2016/FPGA01/FPGA01.json";

    eprintln!("loading chipdb {}", chipdb);
    let db = ChipDb::load(Path::new(chipdb)).expect("load chipdb");
    let mut ctx = Context::new(db);
    eprintln!("loading design {}", design);
    let json = std::fs::read_to_string(design).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    eprintln!("packing");
    packer::pack(&mut ctx, None).expect("pack");

    eprintln!("placing fixed cells via centroid boundary init");
    std::env::set_var("NPNR_OT_INIT", "centroid");
    nextpnr::placer::pipeline::PlacerPipeline::prepare_discrete(&mut ctx, 42)
        .expect("prepare_discrete");

    // Movable cells (those bound by the placer strength after prepare_discrete,
    // excluding fixed cells like IOBs/BUFGs that are locked at chip boundary).
    let idx_to_cell: Vec<CellId> = ctx
        .design
        .iter_alive_cells()
        .filter_map(|(cid, cell)| {
            let strong = cell
                .bel
                .map(|_| matches!(cell.bel_strength, nextpnr::common::PlaceStrength::Locked
                    | nextpnr::common::PlaceStrength::Fixed
                    | nextpnr::common::PlaceStrength::User))
                .unwrap_or(false);
            if strong { None } else { Some(cid) }
        })
        .collect();
    let n = idx_to_cell.len();
    let w = ctx.chipdb().width();
    let h = ctx.chipdb().height();
    let cx = (w as f64 - 1.0) / 2.0;
    let cy = (h as f64 - 1.0) / 2.0;
    let cell_x = vec![cx; n];
    let cell_y = vec![cy; n];
    eprintln!(
        "synthetic pile: {} cells all at ({}, {}) on {}x{} chip",
        n, cx, cy, w, h,
    );

    let type_aware = TypeAwarePlacement::build(&ctx, 0, 0);
    let result = SnapLegalizer
        .legalize(&mut ctx, &idx_to_cell, &cell_x, &cell_y, &type_aware)
        .expect("legalize");

    let rms_disp = (result / n.max(1) as f64).sqrt();
    let post_hpwl = nextpnr::metrics::wirelength::total_hpwl(&ctx);
    eprintln!(
        "DONE: n_cells={} post_legal_HPWL={:.0} rms_disp={:.2} total_sq_disp={:.0}",
        n, post_hpwl, rms_disp, result,
    );
}
