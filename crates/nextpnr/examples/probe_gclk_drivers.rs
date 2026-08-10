//! Find all pips that drive M2_GCLK_B* / M1_GCLK_L_B* wires inside
//! a CLB tile. Tells us whether CLB BEL outputs can reach the GCLK
//! backbone, or only the CLK_RELAY mesh can.

use nextpnr::chipdb::ChipDb;
use nextpnr::read_packed;
use std::path::Path;

fn main() {
    let p = "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin";
    let db = ChipDb::load(Path::new(p)).expect("load");
    let w = db.width();
    let h = db.height();

    // Find an interior CLB.
    let mut clb_tile = -1;
    for tile in 0..(w * h) {
        if db.tile_type_name(tile) == "CLB" {
            let (x, y) = db.tile_xy(tile);
            if x > 20 && y > 20 && x < w - 20 && y < h - 20 {
                clb_tile = tile;
                break;
            }
        }
    }
    assert!(clb_tile >= 0);
    let (cx, cy) = db.tile_xy(clb_tile);
    println!(
        "Sampling CLB tile @ ({},{}) tt_idx={}",
        cx,
        cy,
        db.tile_type_index(clb_tile)
    );

    let tt_idx = db.tile_type_index(clb_tile);
    let tt = db.tile_type_by_index(tt_idx);
    let wires = tt.wires.get();
    let pips = tt.pips.get();

    // Map wire idx -> name.
    let wire_name = |wi: usize| -> String {
        if wi >= wires.len() {
            return "?".into();
        }
        let nid: i32 = unsafe { read_packed!(wires[wi], name) };
        db.constid_str(nid)
            .map(|s| s.to_string())
            .unwrap_or_default()
    };

    // Collect pip drivers per dst wire matching M{1,2}_GCLK*.
    let mut by_dst: Vec<(String, Vec<String>)> = Vec::new();
    for (wi, _w) in wires.iter().enumerate() {
        let nm = wire_name(wi);
        if !nm.starts_with("M1_GCLK") && !nm.starts_with("M2_GCLK") {
            continue;
        }
        let mut srcs = Vec::new();
        for pip in pips.iter() {
            let dst: i32 = unsafe { read_packed!(*pip, dst_wire) };
            if dst as usize == wi {
                let src: i32 = unsafe { read_packed!(*pip, src_wire) };
                srcs.push(wire_name(src as usize));
            }
        }
        by_dst.push((nm, srcs));
    }

    println!("\n=== drivers of M{{1,2}}_GCLK_* wires inside one CLB tile ===");
    for (nm, srcs) in &by_dst {
        println!("  {} <- ({} drivers)", nm, srcs.len());
        for s in srcs.iter().take(8) {
            println!("    {}", s);
        }
        if srcs.len() > 8 {
            println!("    ... +{} more", srcs.len() - 8);
        }
    }

    // Now look at pips that drive *anything* from any BEL output (slice
    // outputs typically have names starting with "CLBLL_*" / "CLBLM_*"
    // or end in "_O" pins). Show them only if their dst is M*_GCLK_*.
    // Actually simpler: dump *all* pip dsts that are M*_GCLK_* wires
    // and bucket their srcs by name prefix.
    use std::collections::HashMap;
    let mut prefix_counts: HashMap<String, usize> = HashMap::new();
    for (_nm, srcs) in &by_dst {
        for s in srcs {
            let prefix = s.split('_').next().unwrap_or("?").to_string();
            *prefix_counts.entry(prefix).or_default() += 1;
        }
    }
    let mut pc: Vec<_> = prefix_counts.into_iter().collect();
    pc.sort_by(|a, b| b.1.cmp(&a.1));
    println!("\n=== driver-name prefix histogram across all M*_GCLK wires ===");
    for (p, c) in pc {
        println!("  {}: {}", p, c);
    }
}
