//! End-to-end routing on a fabric with multi-tile routing nodes.
//!
//! `shared_chipdb.bin` is built by upstream's own himbaechel example generator,
//! so its inter-tile wires are real node shapes, and `placed.json` is a design
//! upstream nextpnr placed on it. Routing that placement with each of our
//! routers exercises node handling on every path that binds, rips up, or
//! searches across a node -- something the single-tile test fabrics cannot.

use nextpnr::chipdb::{parse_constids_inc, BelId, ChipDb};
use nextpnr::common::PlaceStrength;
use nextpnr::context::Context;
use nextpnr::frontend::parse_json;
use nextpnr::netlist::NetId;
use nextpnr::router::common::{collect_sink_wires, resolve_source_wire, source_wire_const_value};
use nextpnr::router::maze::Router1Cfg;
use nextpnr::router::raster::RasterRouterCfg;
use nextpnr::router::router2::Router2Cfg;
use nextpnr::router::{RasterRouter, Router, Router1, Router2};
use std::collections::HashMap;
use std::path::PathBuf;

fn fixture(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests/fixtures/npnr_baseline")
        .join(name)
}

/// The placement upstream nextpnr produced, with no routing.
fn placed_context() -> Context {
    let known = parse_constids_inc(&std::fs::read_to_string(fixture("constids.inc")).unwrap());
    let chipdb = ChipDb::load_with_known_constids(&fixture("shared_chipdb.bin"), &known)
        .expect("the shared chipdb must load with the arch's constids");
    let mut ctx = Context::new(chipdb);

    let json = std::fs::read_to_string(fixture("placed.json")).expect("placed fixture");
    ctx.design = parse_json(&json, &ctx.id_pool).expect("nextpnr output must parse");

    let mut by_name: HashMap<String, BelId> = HashMap::new();
    for bel in ctx.chipdb().bels() {
        let loc = ctx.chipdb().bel_loc(bel);
        by_name.insert(
            format!("X{}Y{}/{}", loc.x, loc.y, ctx.chipdb().bel_name(bel)),
            bel,
        );
    }
    let bel_attr = ctx.id("NEXTPNR_BEL");
    let bindings: Vec<_> = ctx
        .design
        .iter_cell_indices()
        .filter_map(|cell_idx| {
            let name = ctx.design.cell(cell_idx).attrs.get(&bel_attr)?.as_str();
            Some((by_name[&name], cell_idx))
        })
        .collect();
    for (bel, cell_idx) in bindings {
        assert!(ctx.bind_bel(bel, cell_idx, PlaceStrength::Strong));
    }
    ctx
}

/// Nets whose driver and every user were placed. The fixture leaves the
/// top-level port cells unplaced, so nets touching them are not routable here.
fn placed_nets(ctx: &Context) -> Vec<NetId> {
    ctx.design
        .iter_net_indices()
        .filter(|&n| {
            let net = ctx.net(n);
            let placed = |c| ctx.cell(c).bel().is_some();
            net.is_alive()
                && net.num_users() > 0
                && net.driver().is_some_and(|d| d.is_valid() && placed(d.cell))
                && net.users().iter().all(|u| u.is_valid() && placed(u.cell))
        })
        .collect()
}

/// The placed nets not driven from a constant wire.
fn placed_signal_nets(ctx: &Context) -> Vec<NetId> {
    placed_nets(ctx)
        .into_iter()
        .filter(|&n| {
            let source = resolve_source_wire(ctx, n)
                .expect("a placed net resolves its source")
                .expect("a placed net has a source wire");
            source_wire_const_value(ctx, source) == 0
        })
        .collect()
}

/// How much of the placed design a router connected, measured the same way
/// whatever wire identity the router recorded.
struct Coverage {
    nets: usize,
    nets_complete: usize,
    sinks: usize,
    sinks_connected: usize,
    wire_entries: usize,
}

fn coverage(ctx: &Context, nets: &[NetId]) -> Coverage {
    let chipdb = ctx.chipdb();
    let mut c = Coverage {
        nets: nets.len(),
        nets_complete: 0,
        sinks: 0,
        sinks_connected: 0,
        wire_entries: 0,
    };
    for &n in nets {
        let net = ctx.net(n);
        c.wire_entries += net.wires().len();
        // A router that records several members of one node must not hide
        // the member that carries the driving pip behind one that does not.
        let mut tree: HashMap<_, Option<_>> = HashMap::new();
        for (&w, pm) in net.wires() {
            let slot = tree.entry(chipdb.canonical_wire(w)).or_default();
            *slot = slot.or(pm.pip);
        }
        let Some(source) = resolve_source_wire(ctx, n).ok().flatten() else {
            continue;
        };
        let source = chipdb.canonical_wire(source);
        let sinks = collect_sink_wires(ctx, n);
        let mut all = true;
        for sink in sinks {
            c.sinks += 1;
            let mut w = chipdb.canonical_wire(sink);
            let mut steps = 0;
            let connected = loop {
                if w == source {
                    break true;
                }
                match tree.get(&w) {
                    Some(Some(pip)) if steps < 100_000 => {
                        w = chipdb.canonical_wire(chipdb.pip_src_wire(*pip));
                        steps += 1;
                    }
                    _ => break false,
                }
            };
            c.sinks_connected += connected as usize;
            all &= connected;
        }
        c.nets_complete += all as usize;
    }
    c
}

/// Route the placed nets, report coverage, and check that every wire a net
/// records is the one identity its node has and is bound to that net: a node
/// is bound once, not once per tile it crosses.
///
/// `reserves_pins`: the router binds BEL-pin nodes up front, so some
/// bindings belong to no routed tree; there are at most one per placed pin.
fn route_and_check<R: Router>(
    router: &R,
    cfg: &R::Config,
    name: &str,
    reserves_pins: bool,
    select: fn(&Context) -> Vec<NetId>,
) -> Coverage {
    let mut ctx = placed_context();
    let nets = select(&ctx);
    let result = router.route_nets(&mut ctx, cfg, &nets);
    let c = coverage(&ctx, &nets);
    eprintln!(
        "{name}: result={:?} nets {}/{} complete, sinks {}/{} connected, {} wire entries, {} bound wires",
        result.as_ref().err(),
        c.nets_complete,
        c.nets,
        c.sinks_connected,
        c.sinks,
        c.wire_entries,
        ctx.bound_wire_count(),
    );

    let chipdb = ctx.chipdb();
    for &n in &nets {
        for &wire in ctx.net(n).wires().keys() {
            assert_eq!(
                chipdb.canonical_wire(wire),
                wire,
                "{name}: net records node member {wire} instead of its root"
            );
            assert_eq!(
                ctx.wire_binding(wire).map(|(owner, _)| owner),
                Some(n),
                "{name}: wire {wire} is not bound to the net that records it"
            );
        }
    }
    if reserves_pins {
        let pins: usize = ctx
            .design
            .iter_alive_cells()
            .filter(|(_, cell)| cell.bel.is_some())
            .map(|(_, cell)| cell.num_ports())
            .sum();
        assert!(
            ctx.bound_wire_count() <= c.wire_entries + pins,
            "{name}: {} bindings for {} recorded wires and {pins} pins",
            ctx.bound_wire_count(),
            c.wire_entries,
        );
    } else {
        assert_eq!(
            ctx.bound_wire_count(),
            c.wire_entries,
            "{name}: every binding belongs to exactly one recorded wire"
        );
    }
    c
}

#[test]
fn router1_routes_placed_design_on_nodes() {
    // Router1 stops at the first unroutable net, and no constant net routes
    // on this fabric, so routing them would measure only their place in the
    // net order.
    let c = route_and_check(
        &Router1,
        &Router1Cfg::default(),
        "router1",
        false,
        placed_signal_nets,
    );
    assert!(
        c.sinks_connected >= 148,
        "router1 connected {}",
        c.sinks_connected
    );
}

#[test]
fn router2_routes_placed_design_on_nodes() {
    let c = route_and_check(
        &Router2,
        &Router2Cfg::default(),
        "router2",
        false,
        placed_nets,
    );
    assert!(
        c.sinks_connected >= 163,
        "router2 connected {}",
        c.sinks_connected
    );
}

#[test]
fn raster_routes_placed_design_on_nodes() {
    let c = route_and_check(
        &RasterRouter,
        &RasterRouterCfg::default(),
        "raster",
        true,
        placed_nets,
    );
    assert!(
        c.sinks_connected >= 323,
        "raster connected {}",
        c.sinks_connected
    );
}

/// Binding any member of a node binds the node: the binding is visible
/// through every member, and releasing it through any member releases it.
#[test]
fn binding_is_per_node_not_per_member() {
    let mut ctx = placed_context();
    let chipdb = ctx.chipdb();
    let (root, member) = chipdb
        .wires()
        .find_map(|root| {
            let mut member = None;
            chipdb.node_wires_cb(root, |m| member = member.or(Some(m)));
            member.map(|m| (root, m))
        })
        .expect("the example fabric has multi-tile nodes");
    assert_ne!(root, member);
    assert_eq!(chipdb.canonical_wire(member), root);

    let net = ctx.design.iter_net_indices().next().unwrap();
    ctx.try_bind_wire_node(member, net, PlaceStrength::Strong)
        .expect("free node binds");
    assert_eq!(ctx.wire_binding(root).map(|(n, _)| n), Some(net));
    assert_eq!(ctx.wire_binding(member).map(|(n, _)| n), Some(net));
    assert_eq!(ctx.bound_wire_count(), 1, "a node is one binding");

    ctx.unbind_wire(member);
    assert_eq!(ctx.wire_binding(root), None);
    assert_eq!(ctx.bound_wire_count(), 0);
}
