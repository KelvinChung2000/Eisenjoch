//! Shared reporting helpers for placement algorithms.

use std::io::Write;

use crate::context::Context;
use crate::metrics::{total_hpwl, total_line_estimate};
use crate::netlist::CellId;

/// Print the HPWL/line metrics after legalization and return the post-legalization values.
pub fn report_post_legalization(ctx: &Context) -> (f64, f64) {
    let post_hpwl = total_hpwl(ctx);
    let post_line = total_line_estimate(ctx);
    eprintln!("Post-legalization: HPWL={:.0}, line={:.0}", post_hpwl, post_line);
    (post_hpwl, post_line)
}

/// Print the highest-HPWL nets and how concentrated the total HPWL is.
pub fn report_top_net_hpwl(ctx: &Context, top_n: usize) {
    use crate::metrics::wirelength::net_hpwl;

    let mut net_hpwls: Vec<(f64, usize, crate::netlist::NetId)> = Vec::new();
    for (net_id, net) in ctx.design.iter_alive_nets() {
        if net.driver().is_none() || net.num_users() == 0 {
            continue;
        }
        net_hpwls.push((net_hpwl(ctx, net_id), net.num_users(), net_id));
    }
    net_hpwls.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
    let total: f64 = net_hpwls.iter().map(|(h, _, _)| h).sum();
    eprintln!("Per-net HPWL: {} nets, total={:.0}", net_hpwls.len(), total);
    for (i, &(h, f, _)) in net_hpwls.iter().take(top_n).enumerate() {
        eprintln!(
            "  #{}: hpwl={:.0} fanout={} ({:.1}%)",
            i,
            h,
            f,
            if total > 0.0 { h / total * 100.0 } else { 0.0 },
        );
    }
    let mut cumul = 0.0;
    for (i, &(h, _, _)) in net_hpwls.iter().enumerate() {
        cumul += h;
        if cumul > total * 0.5 {
            eprintln!(
                "  50% of HPWL from top {} nets (of {})",
                i + 1,
                net_hpwls.len()
            );
            break;
        }
    }
}

/// Dump continuous and legalized cell positions to CSV.
pub fn dump_position_csv(
    ctx: &Context,
    path: &str,
    idx_to_cell: &[CellId],
    continuous_x: &[f64],
    continuous_y: &[f64],
) -> std::io::Result<usize> {
    let mut f = std::fs::File::create(path)?;
    writeln!(f, "idx,cont_x,cont_y,legal_x,legal_y,disp")?;
    for (i, &cell_id) in idx_to_cell.iter().enumerate() {
        let cell = ctx.design.cell(cell_id);
        let (lx, ly) = if let Some(bel) = cell.bel {
            let loc = ctx.chipdb.bel_loc(bel);
            (loc.x as f64, loc.y as f64)
        } else {
            (continuous_x[i], continuous_y[i])
        };
        let dx = lx - continuous_x[i];
        let dy = ly - continuous_y[i];
        let disp = (dx * dx + dy * dy).sqrt();
        writeln!(
            f,
            "{},{:.3},{:.3},{:.0},{:.0},{:.3}",
            i, continuous_x[i], continuous_y[i], lx, ly, disp
        )?;
    }
    Ok(continuous_x.len())
}

/// Dump placement CSV when the given environment variable is set.
pub fn dump_position_csv_from_env(
    ctx: &Context,
    env_var: &str,
    idx_to_cell: &[CellId],
    continuous_x: &[f64],
    continuous_y: &[f64],
) -> std::io::Result<()> {
    if let Ok(path) = std::env::var(env_var) {
        let count = dump_position_csv(ctx, &path, idx_to_cell, continuous_x, continuous_y)?;
        eprintln!("Dumped {} cell positions to {}", count, path);
    }
    Ok(())
}
