//! Beckmann optimal transport placer.
//!
//! Models placement as a single Kirchhoff system where:
//! - Nets inject flow demand at cell positions (bilinear interpolation)
//! - Pressure drives flow through the pipe network
//! - Cells are Lagrangian particles moving along -grad(P)
//! - Flow interference between nets sharing pipes causes natural spreading
//! - Resistance sigma encodes ALL physics: congestion, interference, timing
//!
//! Uses Anderson-accelerated fixed-point iteration (~20-30 Kirchhoff solves).

mod algorithm;
pub(crate) mod clustering;
pub mod config;
mod demand;
pub(crate) mod network;
mod resistance;

pub use algorithm::place_opt_trans;
pub use config::OptTransPlacerCfg;

use rustc_hash::FxHashSet;

use crate::context::Context;
use crate::netlist::CellId;
use crate::placer::common;
use crate::placer::PlacerError;

#[doc(hidden)]
pub fn debug_lbfgs_smoke_trace() -> Vec<String> {
    use crate::solver::optimizer::LbfgsOptimizer;
    use demand::{accumulate_physical_energy_gradient, build_net_rhs, physical_net_energy, NetSolveInfo};
    use network::{Node, PipeNetwork};

    fn simple_network() -> PipeNetwork {
        PipeNetwork {
            nodes: vec![
                Node {
                    tile_x: 0,
                    tile_y: 0,
                    sub_x: 0,
                    sub_y: 0,
                    pressure: 0.0,
                },
                Node {
                    tile_x: 1,
                    tile_y: 0,
                    sub_x: 0,
                    sub_y: 0,
                    pressure: 0.0,
                },
                Node {
                    tile_x: 0,
                    tile_y: 1,
                    sub_x: 0,
                    sub_y: 0,
                    pressure: 0.0,
                },
                Node {
                    tile_x: 1,
                    tile_y: 1,
                    sub_x: 0,
                    sub_y: 0,
                    pressure: 0.0,
                },
            ],
            pipes: Vec::new(),
            node_pipes: vec![Vec::new(); 4],
            pipe_lookup: Default::default(),
            width: 2,
            height: 2,
            x0: 0,
            y0: 0,
            resolution: 1,
            zero_bel_tiles: 0,
            coarsen: 1,
        }
    }

    fn bowl_network() -> PipeNetwork {
        let mut nodes = Vec::new();
        for y in 0..3 {
            for x in 0..3 {
                nodes.push(Node {
                    tile_x: x,
                    tile_y: y,
                    sub_x: 0,
                    sub_y: 0,
                    pressure: 0.0,
                });
            }
        }
        PipeNetwork {
            nodes,
            pipes: Vec::new(),
            node_pipes: vec![Vec::new(); 9],
            pipe_lookup: Default::default(),
            width: 3,
            height: 3,
            x0: 0,
            y0: 0,
            resolution: 1,
            zero_bel_tiles: 0,
            coarsen: 1,
        }
    }

    fn grid_network(width: i32, height: i32) -> PipeNetwork {
        let mut nodes = Vec::new();
        for y in 0..height {
            for x in 0..width {
                nodes.push(Node {
                    tile_x: x,
                    tile_y: y,
                    sub_x: 0,
                    sub_y: 0,
                    pressure: 0.0,
                });
            }
        }
        PipeNetwork {
            nodes,
            pipes: Vec::new(),
            node_pipes: vec![Vec::new(); (width * height) as usize],
            pipe_lookup: Default::default(),
            width,
            height,
            x0: 0,
            y0: 0,
            resolution: 1,
            zero_bel_tiles: 0,
            coarsen: 1,
        }
    }

    fn instantiate_nets(network: &PipeNetwork, templates: &[NetSolveInfo], pos: &[f64]) -> Vec<NetSolveInfo> {
        let n = pos.len() / 2;
        templates
            .iter()
            .map(|info| {
                let pins = info
                    .pins
                    .iter()
                    .map(|&(px, py, ci_opt)| {
                        if let Some(ci) = ci_opt {
                            (pos[ci], pos[n + ci], Some(ci))
                        } else {
                            (px, py, None)
                        }
                    })
                    .collect();
                NetSolveInfo::from_pins(
                    info.net_id,
                    info.debug_name.clone(),
                    pins,
                    info.pin_is_fixed.clone(),
                    info.has_fixed_pin,
                    network,
                )
            })
            .collect()
    }

    fn scalar_and_grad(
        network: &PipeNetwork,
        pressure: &[f64],
        cfg: &OptTransPlacerCfg,
        templates: &[NetSolveInfo],
        pos: &[f64],
    ) -> (f64, Vec<f64>) {
        let mut energy = 0.0;
        let mut grad = vec![0.0; pos.len()];
        let (grad_x, grad_y) = grad.split_at_mut(pos.len() / 2);
        let nets = instantiate_nets(network, templates, pos);

        for info in &nets {
            let rhs = build_net_rhs(info, network, cfg);
            energy += physical_net_energy(info, &rhs, pressure);
            accumulate_physical_energy_gradient(info, pressure, network, cfg, 1.0, grad_x, grad_y);
        }

        (energy, grad)
    }

    fn run_case(
        name: &str,
        network: PipeNetwork,
        pressure: Vec<f64>,
        nets: Vec<NetSolveInfo>,
        mut pos: Vec<f64>,
        clamp_max: f64,
    ) -> Vec<String> {
        let cfg = OptTransPlacerCfg::default();
        let mut opt = LbfgsOptimizer::new(5);
        let mut out = vec![format!("{name}: start pos={pos:?}")];
        let n = pos.len() / 2;

        for iter in 0..8 {
            let (energy, grad) = scalar_and_grad(&network, &pressure, &cfg, &nets, &pos);
            let step = opt.step(&pos, &grad);
            let directional_derivative: f64 =
                grad.iter().zip(step.iter()).map(|(g, s)| g * s).sum();

            let mut t = 1.0;
            let mut best = None;
            while t >= 1e-6 {
                let mut trial = pos.clone();
                for i in 0..n {
                    trial[i] = (pos[i] + t * step[i]).clamp(0.0, clamp_max);
                    trial[n + i] = (pos[n + i] + t * step[n + i]).clamp(0.0, clamp_max);
                }
                let (trial_energy, _) = scalar_and_grad(&network, &pressure, &cfg, &nets, &trial);
                if trial_energy < energy {
                    best = Some((t, trial, trial_energy));
                    break;
                }
                t *= 0.5;
            }

            let Some((ls, trial, trial_energy)) = best else {
                out.push(format!(
                    "iter {iter}: no descent step found; energy={energy:.6} dirderiv={directional_derivative:.6} grad={grad:?} step={step:?}"
                ));
                return out;
            };

            out.push(format!(
                "iter {iter}: energy={energy:.6}->{trial_energy:.6} ls={ls:.3} pos={pos:?}->{trial:?} dirderiv={directional_derivative:.6}"
            ));
            pos = trial;
        }

        let (final_energy, _) = scalar_and_grad(&network, &pressure, &cfg, &nets, &pos);
        out.push(format!("{name}: final pos={pos:?} energy={final_energy:.6}"));
        out
    }

    let one_net = vec![NetSolveInfo::from_pins(
        crate::netlist::NetId::from_raw(0),
        "one_net",
        vec![(0.0, 0.0, None), (0.8, 0.8, Some(0))],
        vec![true, false],
        false,
        &simple_network(),
    )];

    let two_net = vec![
        NetSolveInfo::from_pins(
            crate::netlist::NetId::from_raw(1),
            "two_net_a",
            vec![(0.0, 0.0, None), (0.8, 0.7, Some(0)), (0.6, 0.9, Some(1))],
            vec![true, false, false],
            false,
            &simple_network(),
        ),
        NetSolveInfo::from_pins(
            crate::netlist::NetId::from_raw(2),
            "two_net_b",
            vec![(1.0, 0.0, None), (0.7, 0.8, Some(2))],
            vec![true, false],
            false,
            &simple_network(),
        ),
    ];

    let bowl_case = vec![
        NetSolveInfo::from_pins(
            crate::netlist::NetId::from_raw(3),
            "bowl_a",
            vec![(1.0, 1.0, None), (1.8, 1.7, Some(0))],
            vec![true, false],
            false,
            &bowl_network(),
        ),
        NetSolveInfo::from_pins(
            crate::netlist::NetId::from_raw(4),
            "bowl_b",
            vec![(1.0, 1.0, None), (0.3, 1.8, Some(1))],
            vec![true, false],
            false,
            &bowl_network(),
        ),
    ];

    let mut out = run_case(
        "one-net",
        simple_network(),
        vec![0.0, 1.0, 2.0, 4.0],
        one_net,
        vec![0.8, 0.8],
        1.0,
    );
    out.extend(run_case(
        "two-net",
        simple_network(),
        vec![0.0, 1.0, 2.0, 4.0],
        two_net,
        vec![0.8, 0.6, 0.7, 0.7, 0.9, 0.8],
        1.0,
    ));
    out.extend(run_case(
        "bowl",
        bowl_network(),
        vec![
            4.0, 2.0, 4.0,
            2.0, 0.0, 2.0,
            4.0, 2.0, 4.0,
        ],
        bowl_case,
        vec![1.8, 0.3, 1.7, 1.8],
        2.0,
    ));
    out.extend(run_case(
        "medium",
        grid_network(5, 5),
        vec![
            8.0, 6.0, 5.0, 6.0, 8.0,
            6.0, 3.0, 1.5, 3.0, 6.0,
            5.0, 1.5, 0.0, 1.5, 5.0,
            6.0, 3.0, 1.5, 3.0, 6.0,
            8.0, 6.0, 5.0, 6.0, 8.0,
        ],
        vec![
            NetSolveInfo::from_pins(
                crate::netlist::NetId::from_raw(5),
                "medium_a",
                vec![(2.0, 2.0, None), (4.2, 4.1, Some(0)), (3.6, 0.8, Some(1))],
                vec![true, false, false],
                false,
                &grid_network(5, 5),
            ),
            NetSolveInfo::from_pins(
                crate::netlist::NetId::from_raw(6),
                "medium_b",
                vec![(2.0, 2.0, None), (0.4, 3.8, Some(2)), (1.1, 0.6, Some(3))],
                vec![true, false, false],
                false,
                &grid_network(5, 5),
            ),
            NetSolveInfo::from_pins(
                crate::netlist::NetId::from_raw(7),
                "medium_c",
                vec![(2.0, 2.0, None), (4.0, 2.6, Some(4))],
                vec![true, false],
                false,
                &grid_network(5, 5),
            ),
        ],
        vec![
            4.2, 3.6, 0.4, 1.1, 4.0,
            4.1, 0.8, 3.8, 0.6, 2.6,
        ],
        4.0,
    ));
    out
}

pub struct PlacerOptTrans;

impl super::Placer for PlacerOptTrans {
    type Config = OptTransPlacerCfg;

    fn place(&self, ctx: &mut Context, cfg: &Self::Config) -> Result<(), PlacerError> {
        place_opt_trans(ctx, cfg)
    }

    fn place_cells(
        &self,
        ctx: &mut Context,
        cfg: &Self::Config,
        cells: &[CellId],
    ) -> Result<(), PlacerError> {
        let target: FxHashSet<CellId> = cells.iter().copied().collect();
        common::with_locked_others(ctx, &target, |ctx| place_opt_trans(ctx, cfg))
    }
}
