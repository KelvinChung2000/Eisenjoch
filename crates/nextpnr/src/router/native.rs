//! Exchange a fixed placement and its physical routing graph with nextpnr-generic.
//! The generic timing model is for connectivity experiments, not timing signoff.
use crate::chipdb::PipId;
use crate::context::Context;
use crate::netlist::NetId;
use crate::router::common::{
    apply_route_plan, collect_routable_nets, resolve_source_wire, unroute_net, validate_all_routed,
    RoutePlan, SinkRoute,
};
use anyhow::{ensure, Context as _, Result};
use std::collections::{HashMap, HashSet};
use std::ffi::OsStr;
use std::io::{BufWriter, Write};
use std::process::{Command, Stdio};

fn export(ctx: &Context, nets: &[NetId], mut out: impl Write) -> Result<()> {
    let db = ctx.chipdb();
    for w in db.wires() {
        let (x, y) = db.tile_xy(w.tile());
        writeln!(out, "W {w} {x} {y}")?;
    }
    for pip in db.pips() {
        let (x, y) = db.tile_xy(pip.tile());
        let dst = db.pip_dst_wire(pip);
        let delay = ctx.delay_ns(ctx.pip(pip).delay().max_delay())
            + ctx.delay_ns(ctx.wire(dst).delay().max_delay());
        writeln!(
            out,
            "P {pip} {} {dst} {x} {y} {delay}",
            db.pip_src_wire(pip)
        )?;
    }
    // The placement is fixed. Only its connected terminals participate in routing.
    for &net in nets {
        let n = ctx.net(net);
        let driver = n.driver_cell_port().context("missing driver")?;
        writeln!(out, "N {}", net.raw())?;
        for pin in std::iter::once(&driver).chain(n.users().iter().filter(|p| p.is_valid())) {
            let bel = ctx.cell(pin.cell).bel().context("unplaced terminal")?;
            let wire = bel.pin_wire(pin.port).context("unmapped BEL pin")?.id();
            let loc = bel.loc();
            let record = (
                net.raw(),
                pin.cell.raw(),
                bel.id().to_string(),
                loc.x,
                loc.y,
                loc.z,
                ctx.name_of(pin.port),
                wire.to_string(),
                pin == &driver,
            );
            writeln!(out, "T {}", serde_json::to_string(&record)?)?;
        }
    }
    out.flush()?;
    Ok(())
}

/// Route a fully placed design with a Python-enabled nextpnr-generic binary.
/// The adapter exports electrical nodes and PIPs from the Rust chipdb.
pub fn route(ctx: &mut Context, binary: &OsStr, seed: u64) -> Result<()> {
    let nets = collect_routable_nets(ctx);
    ensure!(
        ctx.bound_wire_count() == 0,
        "native router2 requires a fresh routing context"
    );
    if nets.is_empty() {
        return Ok(());
    }
    let mut script = tempfile::Builder::new().suffix(".py").tempfile()?;
    script.write_all(include_bytes!("native_router2.py"))?;
    script.flush()?;
    let mut child = Command::new(binary)
        .args(["--router", "router2", "--threads", "1", "--seed"])
        .arg(seed.to_string())
        .arg("--run")
        .arg(script.path())
        .stdin(Stdio::piped())
        .stdout(Stdio::piped())
        .spawn()?;
    let written = export(ctx, &nets, BufWriter::new(child.stdin.take().unwrap()));
    let output = child.wait_with_output()?;
    ensure!(
        output.status.success(),
        "native nextpnr exited with {}",
        output.status
    );
    written?;
    let mut plans = HashMap::new();
    let mut roots = HashSet::new();
    for &net in &nets {
        let source_wire = resolve_source_wire(ctx, net)?.context("missing source")?;
        plans.insert(
            net,
            RoutePlan {
                net,
                source_wire,
                sink_routes: Vec::new(),
            },
        );
    }
    for line in std::str::from_utf8(&output.stdout)?.lines() {
        let Some(record) = line.strip_prefix("ROUTED_WIRE ") else {
            continue;
        };
        let mut fields = record.split_whitespace();
        let raw = fields.next().context("missing net")?.parse()?;
        let net = NetId::from_raw(raw);
        let wire = fields.next().context("missing wire")?;
        let parent = fields.next().context("missing parent")?;
        let plan = plans.get_mut(&net).context("unknown routed net")?;
        if parent == "-" {
            ensure!(
                wire == plan.source_wire.to_string() && roots.insert(net),
                "unexpected route root for net {raw}"
            );
            continue;
        }
        let (tile, index) = parent.split_once(':').context("invalid pip")?;
        let pip = PipId::new(tile.parse()?, index.parse()?);
        let db = ctx.chipdb();
        ensure!(
            (0..db.num_tiles() as i32).contains(&pip.tile())
                && (0..db.tile_type(pip.tile()).pips.get().len() as i32).contains(&pip.index()),
            "unknown pip {parent}"
        );
        let dst = db.pip_dst_wire(pip);
        ensure!(wire == dst.to_string(), "PIP {pip} does not drive {wire}");
        plan.sink_routes.push(SinkRoute {
            sink_wire: dst,
            pips: vec![pip],
        });
    }
    ensure!(roots.len() == nets.len(), "missing native route roots");
    let apply = (|| {
        for &net in &nets {
            apply_route_plan(ctx, &plans[&net])
                .map_err(|w| anyhow::anyhow!("native routes conflict at {w}"))?;
        }
        validate_all_routed(ctx)?;
        Ok(())
    })();
    if apply.is_err() {
        for net in nets {
            unroute_net(ctx, net);
        }
    }
    apply
}
