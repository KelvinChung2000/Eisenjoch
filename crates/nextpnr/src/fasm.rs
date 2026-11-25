//! FASM features of a routed design, named by the chipdb's pip and BEL `extra_data`.

use thiserror::Error;

use crate::chipdb::{BelId, PipFasmFeature, PipId};
use crate::context::Context;
use crate::netlist::{CellInfo, Property};

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum FeatureSetting {
    Enabled,
    Bits { lsb_first: Vec<bool> },
}

#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FasmFeature {
    pub tile: String,
    pub feature: String,
    pub setting: FeatureSetting,
}

#[derive(Debug, Error, PartialEq, Eq)]
pub enum FasmError {
    #[error("Pip {pip} of net {net} has no FASM feature in the chipdb. Regenerate the chipdb with FASM data.")]
    PipWithoutFeature { pip: String, net: String },
    #[error("BEL {bel} of cell {cell} has no FASM prefix in the chipdb. Regenerate the chipdb with FASM data.")]
    BelWithoutPrefix { bel: String, cell: String },
    #[error("Cell {cell} is not placed on a BEL. Place the design before writing FASM.")]
    UnplacedCell { cell: String },
    #[error("Parameter {param} of cell {cell} is an integer with no bit width. Give it as a bit vector in the netlist.")]
    UnsizedParameter { cell: String, param: String },
    #[error(
        "Parameter {param} of cell {cell} has undefined bits '{bits}'. Set every bit to 0 or 1."
    )]
    UndefinedBits {
        cell: String,
        param: String,
        bits: String,
    },
}

/// Features of every bound pip and every placed cell parameter, sorted by tile then feature.
///
/// Fails when the chipdb lacks FASM data, a cell is unplaced or a parameter has no exact bits.
pub fn fasm_features(ctx: &Context) -> Result<Vec<FasmFeature>, FasmError> {
    let mut features = pip_features(ctx)?;
    for (_, cell) in ctx.design.iter_alive_cells() {
        features.extend(cell_features(ctx, cell)?);
    }
    features.sort_by(|a, b| (&a.tile, &a.feature).cmp(&(&b.tile, &b.feature)));
    Ok(features)
}

fn pip_features(ctx: &Context) -> Result<Vec<FasmFeature>, FasmError> {
    let chipdb = ctx.chipdb();
    let mut features = Vec::new();
    for (_, net) in ctx.design.iter_alive_nets() {
        for pip in net.wires.values().filter_map(|binding| binding.pip) {
            let feature =
                chipdb
                    .pip_fasm_feature(pip)
                    .ok_or_else(|| FasmError::PipWithoutFeature {
                        pip: pip_name(ctx, pip),
                        net: ctx.name_of(net.name).to_owned(),
                    })?;
            let PipFasmFeature::Named(feature) = feature else {
                continue;
            };
            features.push(FasmFeature {
                tile: chipdb.tile_name(pip.tile()),
                feature: feature.to_owned(),
                setting: FeatureSetting::Enabled,
            });
        }
    }
    Ok(features)
}

fn pip_name(ctx: &Context, pip: PipId) -> String {
    let chipdb = ctx.chipdb();
    format!(
        "{}/{}->{}",
        chipdb.tile_name(pip.tile()),
        chipdb.wire_name(chipdb.pip_src_tile_wire(pip)),
        chipdb.wire_name(chipdb.pip_dst_tile_wire(pip)),
    )
}

fn cell_features(ctx: &Context, cell: &CellInfo) -> Result<Vec<FasmFeature>, FasmError> {
    let cell_name = ctx.name_of(cell.name);
    let bel = cell.bel.ok_or_else(|| FasmError::UnplacedCell {
        cell: cell_name.to_owned(),
    })?;
    let prefix = bel_prefix(ctx, bel, cell_name)?;
    let tile = ctx.chipdb().tile_name(bel.tile());
    let mut features = Vec::new();
    for (param, value) in &cell.params {
        let param = ctx.name_of(*param);
        let feature = |name: String, setting: FeatureSetting| FasmFeature {
            tile: tile.clone(),
            feature: format!("{prefix}.{name}"),
            setting,
        };
        match value {
            Property::Int(_) => {
                return Err(FasmError::UnsizedParameter {
                    cell: cell_name.to_owned(),
                    param: param.to_owned(),
                });
            }
            Property::String(text) if is_undefined_bit_vector(text) => {
                return Err(FasmError::UndefinedBits {
                    cell: cell_name.to_owned(),
                    param: param.to_owned(),
                    bits: text.clone(),
                });
            }
            Property::String(text) => {
                features.push(feature(format!("{param}.{text}"), FeatureSetting::Enabled));
            }
            Property::BitVector(bits) if bits.is_empty() => {
                return Err(FasmError::UnsizedParameter {
                    cell: cell_name.to_owned(),
                    param: param.to_owned(),
                });
            }
            Property::BitVector(bits) if !bits.chars().all(|c| matches!(c, '0' | '1')) => {
                return Err(FasmError::UndefinedBits {
                    cell: cell_name.to_owned(),
                    param: param.to_owned(),
                    bits: bits.clone(),
                });
            }
            Property::BitVector(bits) if bits == "1" => {
                features.push(feature(param.to_owned(), FeatureSetting::Enabled));
            }
            Property::BitVector(bits) if bits == "0" => {}
            Property::BitVector(bits) => {
                let lsb_first = bits.chars().rev().map(|bit| bit == '1').collect();
                features.push(feature(
                    param.to_owned(),
                    FeatureSetting::Bits { lsb_first },
                ));
            }
        }
    }
    Ok(features)
}

fn bel_prefix<'a>(ctx: &'a Context, bel: BelId, cell: &str) -> Result<&'a str, FasmError> {
    let chipdb = ctx.chipdb();
    chipdb
        .bel_fasm_prefix(bel)
        .ok_or_else(|| FasmError::BelWithoutPrefix {
            bel: format!("{}/{}", chipdb.tile_name(bel.tile()), chipdb.bel_name(bel)),
            cell: cell.to_owned(),
        })
}

fn is_undefined_bit_vector(text: &str) -> bool {
    !text.is_empty()
        && text
            .chars()
            .all(|c| matches!(c, '0' | '1' | 'x' | 'X' | 'z' | 'Z'))
        && text.chars().any(|c| !matches!(c, '0' | '1'))
}
