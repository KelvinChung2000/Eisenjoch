//! Global nets and the dedicated global network come from the chipdb's own
//! global-buffer flag (himbaechel `BEL_FLAG_GLOBAL`) and its connectivity,
//! never from an arch's wire names.
//!
//! The example fabric has a real clock tree: pad `IO0` in tile (1, 0) drives
//! `GCLK_OUT`, which is chained tile to tile through `CLK`/`CLK_PREV` nodes.
//! Its generator flags no BEL global, so the test sets the flag on `IO0` in a
//! copy of the database and checks what follows from it.

use nextpnr::chipdb::{BelDataPod, BelId, ChipDb, WireId};
use nextpnr::common::PlaceStrength;
use nextpnr::context::Context;
use nextpnr::netlist::PortType;
use nextpnr::router::common::is_global_net;
use std::path::PathBuf;

fn example_bytes() -> Vec<u8> {
    let path = PathBuf::from(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/example.bin");
    std::fs::read(path).expect("example chipdb")
}

fn clock_pad(db: &ChipDb) -> BelId {
    db.bels()
        .find(|&b| {
            let loc = db.bel_loc(b);
            (loc.x, loc.y) == (1, 0) && db.bel_name(b) == "IO0"
        })
        .expect("the example fabric's clock pad")
}

/// The example chipdb with `IO0` at (1, 0) flagged as a global buffer.
fn chipdb_with_global_pad() -> ChipDb {
    let mut bytes = example_bytes();
    let plain = unsafe { ChipDb::from_bytes(&bytes) }.unwrap();
    // Locate the BEL's `flags` field in the file: the root pointer at offset 0
    // gives the chip info's file offset, and the mapped addresses are
    // relative to the same base.
    let root = i32::from_le_bytes(bytes[0..4].try_into().unwrap()) as usize;
    let base = plain.chip_info() as *const _ as usize - root;
    let bel = plain.bel_info(clock_pad(&plain)) as *const BelDataPod as usize;
    let flags_at = bel - base + 12; // name(4) + type(4) + z(2) + padding(2)
    let flags = u32::from_le_bytes(bytes[flags_at..flags_at + 4].try_into().unwrap());
    bytes[flags_at..flags_at + 4].copy_from_slice(&(flags | BelDataPod::FLAG_GLOBAL).to_le_bytes());
    drop(plain);
    unsafe { ChipDb::from_bytes(&bytes) }.unwrap()
}

#[test]
fn unflagged_chipdb_has_no_global_network() {
    let db = unsafe { ChipDb::from_bytes(&example_bytes()) }.unwrap();
    assert!(db.bels().all(|b| !db.bel_is_global(b)));
    assert!(db.global_network().is_empty());
}

#[test]
fn global_network_is_what_only_global_buffers_drive() {
    let db = chipdb_with_global_pad();
    let pad = clock_pad(&db);
    assert!(db.bel_is_global(pad));
    // BEL data is per tile type, so the flag marks that BEL in every tile of
    // the type: what an arch's generator does for all its global buffers.
    let flagged: Vec<BelId> = db.bels().filter(|&b| db.bel_is_global(b)).collect();
    assert!(flagged.len() > 1);
    for &b in &flagged {
        assert_eq!(db.tile_type_index(b.tile()), db.tile_type_index(pad.tile()));
        assert_eq!(db.bel_name(b), "IO0");
    }

    let net = db.global_network();
    let seed = db
        .bel_info(pad)
        .pins
        .get()
        .iter()
        .map(|p| db.bel_pin_fields(p))
        .find(|&(_, _, dir)| dir == PortType::Out as i32)
        .map(|(_, w, _)| db.canonical_wire(WireId::new(pad.tile(), w)))
        .expect("the pad has an output");
    assert!(net.contains(&seed));

    // The clock tree reaches across the fabric, not just the pad's tile.
    let tiles: std::collections::HashSet<i32> = net
        .iter()
        .flat_map(|&w| {
            let mut t = Vec::new();
            db.node_members_cb(w, |m| t.push(m.tile()));
            t
        })
        .collect();
    assert!(tiles.len() > 10, "global network spans {} tiles", tiles.len());

    // Dedicated means dedicated: nothing outside the network drives it, and
    // every node it leaves for is also driven from elsewhere.
    for &w in net {
        if w == seed {
            continue;
        }
        db.pips_uphill_cb(w, |pip| {
            assert!(net.contains(&db.pip_src_wire(pip)), "{w} driven from outside");
        });
    }
    for &w in net {
        db.pips_downhill_cb(w, |pip| {
            let dst = db.pip_dst_wire(pip);
            if !net.contains(&dst) {
                let mut foreign = false;
                db.pips_uphill_cb(dst, |up| foreign |= !net.contains(&db.pip_src_wire(up)));
                assert!(foreign, "{dst} is only driven by the network but left out");
            }
        });
    }
}

#[test]
fn a_net_driven_by_a_global_buffer_is_global() {
    let mut ctx = Context::new(chipdb_with_global_pad());
    let pad = clock_pad(ctx.chipdb());
    assert!(ctx.bel_global_buf(pad));

    let cell = ctx.design.add_cell(ctx.id("clk_buf"), ctx.id("IOB"));
    let net = ctx.design.add_net(ctx.id("clk"));
    let port = ctx.id("O");
    ctx.design.cell_edit(cell).add_port(port, PortType::Out);
    ctx.design.cell_edit(cell).set_port_net(port, Some(net), None);
    ctx.design.net_edit(net).set_driver(cell, port);
    assert!(!is_global_net(&ctx, net), "an unplaced driver is not on a global buffer");
    assert!(ctx.bind_bel(pad, cell, PlaceStrength::Strong));
    assert!(is_global_net(&ctx, net));
}
