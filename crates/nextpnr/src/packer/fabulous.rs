//! Maps `synth_fabulous` LUTn and LUTFF cells onto the FABULOUS_LC BELs of a
//! FABulous `himbaechel_db` chipdb. A flop merges into the LUT that alone drives
//! its D input and requests no other BEL, else into a pass-through LC, and sets
//! FF to register O. Unused EN and SR tie to VCC and GND; a constant pin with no
//! GND0 or VCC0 pip into it gets its own constant LC.

use std::collections::BTreeMap;

use super::helpers::{connect_port, disconnect_port};
use super::{BindingFault, PackerError};
use crate::chipdb::BelId;
use crate::common::{IdString, PlaceStrength};
use crate::context::{BelPin, Context};
use crate::netlist::{CellId, NetId, PortType, Property};

pub const UARCH: &str = "FABulous";
const LC_TYPE: &str = "FABULOUS_LC";
const LC_INIT_BITS: usize = 16;
const PASS_THROUGH_I0_INIT: &str = "1010101010101010";
const GND_NET: &str = "$PACKER_GND_NET";
const VCC_NET: &str = "$PACKER_VCC_NET";

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct BelRequest {
    bel: BelId,
    strength: PlaceStrength,
}

type BelRequests = BTreeMap<CellId, BelRequest>;

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
enum SyncControl {
    Unused,
    Reset,
    Set,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
struct FlopVariant {
    enable: bool,
    control: SyncControl,
}

impl FlopVariant {
    fn from_cell_type(cell_type: &str) -> Option<Self> {
        let (enable, control) = match cell_type {
            "LUTFF" => (false, SyncControl::Unused),
            "LUTFF_E" => (true, SyncControl::Unused),
            "LUTFF_SR" => (false, SyncControl::Reset),
            "LUTFF_SS" => (false, SyncControl::Set),
            "LUTFF_ESR" => (true, SyncControl::Reset),
            "LUTFF_ESS" => (true, SyncControl::Set),
            _ => return None,
        };
        Some(Self { enable, control })
    }
}

fn lut_inputs(cell_type: &str) -> Option<usize> {
    match cell_type {
        "LUT1" => Some(1),
        "LUT2" => Some(2),
        "LUT3" => Some(3),
        "LUT4" => Some(4),
        _ => None,
    }
}

/// Merge flops into LCs, retype every LUT as FABULOUS_LC and bind each cell that a `BEL="X0Y14.B"`
/// attribute or an earlier binding places. A bound flop merges only into an LUT that requests no
/// other BEL; the LC inherits the flop's BEL. Fails on a LUT or LUTFF type this fabric's LC cannot
/// implement, a malformed INIT, a bad BEL attribute or an illegal binding.
pub fn pack_fabulous(ctx: &mut Context) -> Result<(), PackerError> {
    let mut requests = take_bel_requests(ctx)?;
    let mut flops = Vec::new();
    let mut luts = Vec::new();
    for (id, cell) in ctx.design.iter_alive_cells() {
        let cell_type = ctx.name_of(cell.cell_type);
        if let Some(variant) = FlopVariant::from_cell_type(cell_type) {
            flops.push((id, variant));
        } else if let Some(inputs) = lut_inputs(cell_type) {
            luts.push((id, inputs));
        } else if cell_type.starts_with("LUT") {
            return Err(PackerError::UnsupportedCellType(format!(
                "{cell_type} (cell {}) has no FABULOUS_LC mapping",
                ctx.name_of(cell.name)
            )));
        }
    }
    for (lut, inputs) in luts {
        retype_lut(ctx, lut, inputs)?;
    }
    for (flop, variant) in flops {
        merge_flop(ctx, flop, variant, &mut requests)?;
    }
    bind_requests(ctx, requests)?;
    drive_far_constants(ctx, GND_NET, '0')?;
    drive_far_constants(ctx, VCC_NET, '1')?;
    Ok(())
}

fn drive_far_constants(
    ctx: &mut Context,
    net_name: &str,
    init_bit: char,
) -> Result<(), PackerError> {
    let net = constant_net(ctx, net_name)?;
    let users: Vec<(CellId, IdString)> = ctx
        .design
        .net(net)
        .users()
        .iter()
        .filter(|user| user.is_connected())
        .map(|user| (user.cell, user.port))
        .collect();
    if users.is_empty() {
        return Ok(());
    }
    let const_id = driven_const_value(ctx, net)?;
    let far: Vec<(CellId, IdString)> = users
        .into_iter()
        .filter(|&(cell, port)| !reaches_constant_locally(ctx, cell, port, const_id))
        .collect();
    let init = ctx.id("INIT");
    let o = ctx.id("O");
    let lc_type = ctx.id(LC_TYPE);
    for (index, (cell, port)) in far.into_iter().enumerate() {
        let lc = ctx
            .design
            .add_cell(ctx.id(&format!("{net_name}$lc{index}")), lc_type);
        ctx.design
            .cell_edit(lc)
            .set_param(
                init,
                Property::bit_vector(init_bit.to_string().repeat(LC_INIT_BITS)),
            )
            .add_port(o, PortType::Out);
        let driven = ctx
            .design
            .add_net(ctx.id(&format!("{net_name}$lc{index}$O")));
        connect_port(ctx, lc, o, driven);
        disconnect_port(ctx, cell, port);
        connect_port(ctx, cell, port, driven);
    }
    Ok(())
}

fn driven_const_value(ctx: &Context, net: NetId) -> Result<i32, PackerError> {
    let net_name = ctx.name_of(ctx.design.net(net).name).to_owned();
    let driver =
        ctx.design.net(net).driver().ok_or_else(|| {
            PackerError::Generic(format!("Constant net {net_name} has no driver."))
        })?;
    let bel = ctx.design.cell(driver.cell).bel.ok_or_else(|| {
        PackerError::Generic(format!(
            "The driver of constant net {net_name} is unplaced. The chipdb needs GND_DRV and VCC_DRV BELs."
        ))
    })?;
    let wire = ctx
        .bel_pin_wire(BelPin::new(bel, driver.port))
        .ok_or_else(|| {
            PackerError::Generic(format!(
                "The driver of constant net {net_name} has no output wire."
            ))
        })?;
    let chipdb = ctx.chipdb();
    match chipdb.wire_info(wire.id()).const_value {
        0 => Err(PackerError::Generic(format!(
            "Constant net {net_name} is driven onto wire {}, which has no const_value.",
            chipdb.wire_name(wire.id())
        ))),
        value => Ok(chipdb.constid_canon(value)),
    }
}

fn reaches_constant_locally(ctx: &Context, cell: CellId, port: IdString, const_id: i32) -> bool {
    let info = ctx.design.cell(cell);
    let bels: Vec<BelId> = match info.bel {
        Some(bel) => vec![bel],
        None => ctx
            .bels_for_bucket(info.cell_type)
            .map(|bel| bel.id())
            .collect(),
    };
    let chipdb = ctx.chipdb();
    bels.into_iter().all(|bel| {
        ctx.bel_pin_wire(BelPin::new(bel, port))
            .is_some_and(|wire| {
                let mut found = false;
                chipdb.pips_uphill_cb(wire.id(), |pip| {
                    let source = chipdb.wire_info(chipdb.pip_src_wire(pip)).const_value;
                    found |= chipdb.constid_canon(source) == const_id;
                });
                found
            })
    })
}

fn packed_type(cell_type: &str) -> &str {
    if lut_inputs(cell_type).is_some() || FlopVariant::from_cell_type(cell_type).is_some() {
        LC_TYPE
    } else {
        cell_type
    }
}

fn bel_location(ctx: &Context, bel: BelId) -> String {
    let chipdb = ctx.chipdb();
    format!("{}.{}", chipdb.tile_name(bel.tile()), chipdb.bel_name(bel))
}

fn illegal_binding(ctx: &Context, cell: CellId, bel: BelId, fault: BindingFault) -> PackerError {
    PackerError::IllegalBinding {
        cell: ctx.name_of(ctx.design.cell(cell).name).to_owned(),
        bel: bel_location(ctx, bel),
        fault,
    }
}

fn find_bel_attribute(ctx: &Context, cell: CellId, location: &str) -> Result<BelId, PackerError> {
    let cell_name = ctx.name_of(ctx.design.cell(cell).name);
    let (tile, bel_name) = location.split_once('.').ok_or_else(|| {
        PackerError::Generic(format!(
            "Cell {cell_name} has BEL attribute '{location}'. Write it as <tile>.<bel>, for example X0Y14.B."
        ))
    })?;
    let chipdb = ctx.chipdb();
    chipdb
        .bels()
        .find(|&bel| chipdb.tile_name(bel.tile()) == tile && chipdb.bel_name(bel) == bel_name)
        .ok_or_else(|| {
            PackerError::Generic(format!(
                "Cell {cell_name} names BEL {location}, which the chipdb does not have."
            ))
        })
}

/// Collect every `BEL` attribute and every LUT or LUTFF binding, then unbind them until packing ends.
///
/// Fails on a malformed or unknown BEL attribute, an attribute that names a BEL other than the
/// cell's binding, or a BEL whose type is not the type the cell packs to.
fn take_bel_requests(ctx: &mut Context) -> Result<BelRequests, PackerError> {
    let bel_attr = ctx.id("BEL");
    let candidates: Vec<(CellId, Option<String>, Option<BelRequest>)> = ctx
        .design
        .iter_alive_cells()
        .filter_map(|(id, cell)| {
            let attr = cell.attrs.get(&bel_attr).map(Property::as_str);
            let repacked = packed_type(ctx.name_of(cell.cell_type)) == LC_TYPE;
            let bound = cell
                .bel
                .filter(|_| repacked || attr.is_some())
                .map(|bel| BelRequest {
                    bel,
                    strength: cell.bel_strength,
                });
            (attr.is_some() || bound.is_some()).then_some((id, attr, bound))
        })
        .collect();
    let mut requests = BelRequests::new();
    for (cell, attr, bound) in candidates {
        let request = match attr {
            Some(location) => {
                let bel = find_bel_attribute(ctx, cell, &location)?;
                match bound {
                    Some(bound) if bound.bel != bel => {
                        let fault = BindingFault::AlreadyBound(bel_location(ctx, bound.bel));
                        return Err(illegal_binding(ctx, cell, bel, fault));
                    }
                    Some(bound) => bound,
                    None => BelRequest {
                        bel,
                        strength: PlaceStrength::User,
                    },
                }
            }
            None => bound.expect("collected for its attribute or its binding"),
        };
        let packed = packed_type(ctx.name_of(ctx.design.cell(cell).cell_type));
        let bel_type = ctx.chipdb().bel_type(request.bel);
        if bel_type != packed {
            let fault = BindingFault::TypeMismatch {
                bel_type: bel_type.to_owned(),
                packed_type: packed.to_owned(),
            };
            return Err(illegal_binding(ctx, cell, request.bel, fault));
        }
        requests.insert(cell, request);
    }
    for (&cell, request) in &requests {
        if ctx.design.cell(cell).bel.is_some() {
            ctx.unbind_bel(request.bel);
        }
    }
    Ok(requests)
}

/// Bind every surviving cell to its requested BEL.
///
/// Fails when another cell already holds the BEL, such as two LCs that request one BEL.
fn bind_requests(ctx: &mut Context, requests: BelRequests) -> Result<(), PackerError> {
    for (cell, request) in requests {
        if let Some(holder) = ctx.conflicting_bel_cell(request.bel) {
            let holder = ctx.name_of(ctx.design.cell(holder).name).to_owned();
            return Err(illegal_binding(
                ctx,
                cell,
                request.bel,
                BindingFault::Occupied(holder),
            ));
        }
        assert!(
            ctx.bind_bel(request.bel, cell, request.strength),
            "a free chipdb BEL accepts the binding"
        );
    }
    Ok(())
}

fn merge_flop(
    ctx: &mut Context,
    flop: CellId,
    variant: FlopVariant,
    requests: &mut BelRequests,
) -> Result<(), PackerError> {
    let [d, o, clk, en, sr, e, r, s, i0] =
        ["D", "O", "CLK", "EN", "SR", "E", "R", "S", "I0"].map(|name| ctx.id(name));
    let flop_name = ctx.design.cell(flop).name;
    let d_net = ctx.design.cell(flop).port_net(d).ok_or_else(|| {
        PackerError::Generic(format!(
            "Flip-flop {} has no D net.",
            ctx.name_of(flop_name)
        ))
    })?;
    let flop_request = requests.remove(&flop);
    let absorbing_lut =
        sole_lut_driver(ctx, d_net).filter(|lut| match (requests.get(lut), flop_request) {
            (Some(lut_request), Some(flop_request)) => lut_request.bel == flop_request.bel,
            _ => true,
        });
    let lc = match absorbing_lut {
        Some(lut) => {
            disconnect_port(ctx, lut, o);
            disconnect_port(ctx, flop, d);
            let d_name = ctx.design.net(d_net).name;
            ctx.design.remove_net(d_name);
            lut
        }
        None => {
            let lc_name = ctx.id(&format!("{}$lc", ctx.name_of(flop_name)));
            let lc_type = ctx.id(LC_TYPE);
            let lc = ctx.design.add_cell(lc_name, lc_type);
            let init = ctx.id("INIT");
            ctx.design
                .cell_edit(lc)
                .set_param(init, Property::bit_vector(PASS_THROUGH_I0_INIT));
            move_port(ctx, flop, d, lc, i0, PortType::In);
            lc
        }
    };
    if let Some(request) = flop_request {
        requests.entry(lc).or_insert(request);
    }
    let ff = ctx.id("FF");
    ctx.design
        .cell_edit(lc)
        .set_param(ff, Property::bit_vector("1"));
    move_port(ctx, flop, clk, lc, clk, PortType::In);
    move_port(ctx, flop, o, lc, o, PortType::Out);
    if variant.enable {
        move_port(ctx, flop, e, lc, en, PortType::In);
    } else {
        tie(ctx, lc, en, VCC_NET)?;
    }
    match variant.control {
        SyncControl::Unused => tie(ctx, lc, sr, GND_NET)?,
        SyncControl::Reset => move_port(ctx, flop, r, lc, sr, PortType::In),
        SyncControl::Set => {
            move_port(ctx, flop, s, lc, sr, PortType::In);
            let set_noreset = ctx.id("SET_NORESET");
            ctx.design
                .cell_edit(lc)
                .set_param(set_noreset, Property::bit_vector("1"));
        }
    }
    ctx.design.remove_cell(flop_name);
    Ok(())
}

fn sole_lut_driver(ctx: &Context, net: NetId) -> Option<CellId> {
    let info = ctx.design.net(net);
    let driver = info.driver()?;
    let is_lc = ctx.name_of(ctx.design.cell(driver.cell).cell_type) == LC_TYPE;
    let users = info
        .users()
        .iter()
        .filter(|user| user.is_connected())
        .count();
    (is_lc && ctx.name_of(driver.port) == "O" && users == 1).then_some(driver.cell)
}

fn move_port(
    ctx: &mut Context,
    from: CellId,
    from_port: IdString,
    to: CellId,
    to_port: IdString,
    port_type: PortType,
) {
    let Some(net) = ctx.design.cell(from).port_net(from_port) else {
        return;
    };
    disconnect_port(ctx, from, from_port);
    ctx.design.cell_edit(to).add_port(to_port, port_type);
    connect_port(ctx, to, to_port, net);
}

fn constant_net(ctx: &Context, net_name: &str) -> Result<NetId, PackerError> {
    ctx.design.net_by_name(ctx.id(net_name)).ok_or_else(|| {
        PackerError::Generic(format!(
            "Constant net {net_name} is missing. Run pack_constants first."
        ))
    })
}

fn tie(ctx: &mut Context, cell: CellId, port: IdString, net_name: &str) -> Result<(), PackerError> {
    let net = constant_net(ctx, net_name)?;
    ctx.design.cell_edit(cell).add_port(port, PortType::In);
    connect_port(ctx, cell, port, net);
    Ok(())
}

fn retype_lut(ctx: &mut Context, lut: CellId, inputs: usize) -> Result<(), PackerError> {
    let init = ctx.id("INIT");
    let cell = ctx.design.cell(lut);
    let cell_name = ctx.name_of(cell.name).to_owned();
    if let Some(param) = cell.params.keys().find(|&&key| key != init) {
        return Err(PackerError::Generic(format!(
            "LUT {cell_name} has parameter {}, which FABULOUS_LC does not configure.",
            ctx.name_of(*param)
        )));
    }
    let width = 1 << inputs;
    let msb_first = match cell.params.get(&init) {
        Some(Property::BitVector(bits))
            if bits.len() == width && bits.chars().all(|bit| matches!(bit, '0' | '1')) =>
        {
            bits.clone()
        }
        other => {
            return Err(PackerError::Generic(format!(
                "LUT {cell_name} has INIT {other:?}. Give it as a {width}-bit 0/1 vector."
            )));
        }
    };
    let expanded = msb_first.repeat(LC_INIT_BITS / width);
    let lc_type = ctx.id(LC_TYPE);
    ctx.design
        .cell_edit(lut)
        .set_type(lc_type)
        .set_param(init, Property::bit_vector(expanded));
    Ok(())
}
