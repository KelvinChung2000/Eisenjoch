//! Measure or refine a routed LUT6/IOB combinational design's modeled delay.
//! Usage: refine_routed_timing <chipdb> <netlist> <checkpoint> [<output> <passes>]
//! Use the same chipdb and unpacked netlist as the original place-and-route run.

use anyhow::{ensure, Context as _, Result};
use nextpnr::{
    checkpoint::{restore, save, Checkpoint},
    chipdb::ChipDb,
    context::Context,
    frontend::parse_json,
    packer,
    router::timing_refine::refine,
};
use std::path::Path;

fn main() -> Result<()> {
    let args: Vec<_> = std::env::args_os().skip(1).collect();
    ensure!(
        matches!(args.len(), 3 | 5),
        "usage: refine_routed_timing <chipdb> <netlist> <checkpoint> [<output> <passes>]"
    );
    let mut ctx = Context::new(ChipDb::load(Path::new(&args[0]))?);
    ctx.design = parse_json(&std::fs::read_to_string(&args[1])?, &ctx.id_pool)?;
    packer::pack(&mut ctx, None)?;
    let cp = Checkpoint::load_from_file(Path::new(&args[2]))?;
    let report = restore(&mut ctx, &cp)?;
    ensure!(
        report.cells_to_place.is_empty() && report.nets_to_route.is_empty(),
        "checkpoint is incomplete"
    );

    let passes = match args.get(4) {
        Some(value) => value.to_str().context("invalid pass count")?.parse()?,
        None => 0,
    };
    for (pass, result) in refine(&mut ctx, passes)?.into_iter().enumerate() {
        println!(
            "pass={pass} shortened_nets={} total_ps={} logic_ps={} route_ps={}",
            result.shortened_nets,
            result.total_ps(),
            result.logic_ps,
            result.route_ps
        );
    }
    if let Some(output) = args.get(3) {
        save(&ctx, Path::new(output))?;
    }
    Ok(())
}
