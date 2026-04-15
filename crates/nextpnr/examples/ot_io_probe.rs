use nextpnr::chipdb::{BelId, ChipDb};
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::packer;
use std::collections::BTreeMap;
use std::path::Path;

fn main() {
    let chipdb = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_exact.bin";
    let design = "/home/kelvin/side-project/eisenjoch/benchmark/output/stereovision3.json";

    let db = ChipDb::load(Path::new(chipdb)).expect("load chipdb");
    let mut ctx = Context::new(db);
    let json = std::fs::read_to_string(design).expect("read design");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("parse design");
    packer::pack(&mut ctx, None).expect("pack");

    let w = ctx.chipdb().width();
    let h = ctx.chipdb().height();
    println!("chip: {}x{}", w, h);

    // Find bucket IDs present
    let mut buckets: BTreeMap<String, usize> = BTreeMap::new();
    for (_, cell) in ctx.design.iter_alive_cells() {
        let b = ctx.name_of(ctx.resolve_bucket(cell.cell_type)).to_string();
        *buckets.entry(b).or_insert(0) += 1;
    }
    for (b, c) in &buckets {
        println!("bucket {}: {} cells", b, c);
    }

    // Find the IOB bucket id
    let iob_bucket = {
        let mut found = None;
        for (_, cell) in ctx.design.iter_alive_cells() {
            let name = ctx.name_of(ctx.resolve_bucket(cell.cell_type)).to_string();
            if name == "IOB" {
                found = Some(ctx.resolve_bucket(cell.cell_type));
                break;
            }
        }
        found.expect("no IOB cells")
    };

    // Enumerate all IOB BELs
    let mut iob_bels: Vec<(i32, i32, BelId)> = Vec::new();
    for bel in ctx.bels_for_bucket(iob_bucket) {
        let loc = bel.loc();
        iob_bels.push((loc.x, loc.y, bel.id()));
    }
    println!("\ntotal IOB BELs in chipdb: {}", iob_bels.len());

    // Distribution by edge
    let mut by_edge: BTreeMap<&str, usize> = BTreeMap::new();
    let ew = (w as f64 * 0.05).round() as i32;
    let eh = (h as f64 * 0.05).round() as i32;
    for (x, y, _) in &iob_bels {
        let left = *x <= ew;
        let right = *x >= w - ew;
        let bottom = *y <= eh;
        let top = *y >= h - eh;
        let mut found = false;
        if left { *by_edge.entry("left").or_insert(0) += 1; found = true; }
        if right { *by_edge.entry("right").or_insert(0) += 1; found = true; }
        if bottom { *by_edge.entry("bottom").or_insert(0) += 1; found = true; }
        if top { *by_edge.entry("top").or_insert(0) += 1; found = true; }
        if !found { *by_edge.entry("interior").or_insert(0) += 1; }
    }
    println!("\nIOB BEL distribution (within 5% of edge):");
    for (e, c) in &by_edge {
        println!("  {:<8} {}", e, c);
    }

    // Exact columns/rows
    let mut by_col: BTreeMap<i32, usize> = BTreeMap::new();
    let mut by_row: BTreeMap<i32, usize> = BTreeMap::new();
    for (x, y, _) in &iob_bels {
        *by_col.entry(*x).or_insert(0) += 1;
        *by_row.entry(*y).or_insert(0) += 1;
    }
    println!("\ndistinct X (columns) with IOB BELs: {}", by_col.len());
    for (x, c) in by_col.iter().take(20) {
        println!("  x={:>4} count={}", x, c);
    }
    println!("\ndistinct Y (rows) with IOB BELs: {}", by_row.len());
    for (y, c) in by_row.iter().take(20) {
        println!("  y={:>4} count={}", y, c);
    }

    // How many IOBs needed for this design?
    let n_iobs_design: usize = ctx.design.iter_alive_cells()
        .filter(|(_, c)| ctx.name_of(ctx.resolve_bucket(c.cell_type)).to_string() == "IOB")
        .count();
    println!("\ndesign needs {} IOBs", n_iobs_design);
}
