use nextpnr::chipdb::ChipDb;
use nextpnr::context::Context;
use std::collections::BTreeMap;
use std::path::Path;

fn main() {
    let db = ChipDb::load(Path::new(
        "/home/kelvin/side-project/eisenjoch/chip_database/xc7_large.bin",
    ))
    .expect("load");
    let ctx = Context::new(db);
    let cdb = ctx.chipdb();
    println!("grid: {}x{}", cdb.width(), cdb.height());

    // Histogram tile type name at edges
    let mut by_name: BTreeMap<String, usize> = BTreeMap::new();
    for tile in 0..(cdb.width() * cdb.height()) {
        let name = cdb.tile_type_name(tile).to_string();
        *by_name.entry(name).or_insert(0) += 1;
    }
    println!("Tile types in chipdb:");
    for (n, c) in &by_name {
        println!("  {n} : {c}");
    }

    // Print tile types at the failing source coords from probe_h_vs_true
    println!("\nFailing source tiles:");
    for (sx, sy) in [(0i32, 183), (0, 152), (309, 119), (0, 98)] {
        let t = sy * cdb.width() + sx;
        println!(
            "  ({sx},{sy}): type_idx={} name='{}'",
            cdb.tile_type_index(t),
            cdb.tile_type_name(t)
        );
    }

    // And tile types at sample destinations
    println!("\nFailing dest tiles:");
    for (sx, sy) in [
        (130i32, 118),
        (148, 102),
        (199, 90),
        (104, 105),
        (148, 113),
        (189, 117),
        (131, 106),
        (310, 111),
    ] {
        let t = sy * cdb.width() + sx;
        println!(
            "  ({sx},{sy}): type_idx={} name='{}'",
            cdb.tile_type_index(t),
            cdb.tile_type_name(t)
        );
    }
}
