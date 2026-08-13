//! Yosys JSON netlist parser.

use crate::common::IdStringPool;
use crate::netlist::{Design, NetId, PortType};
use anyhow::{bail, Context, Result};
use rustc_hash::FxHashMap;
use serde_json::Value;

use super::helpers::{
    apply_net_names, collect_bit_indices, connect_port_to_net, create_constant_driver,
    create_constant_net, infer_port_direction, parse_bit_value, parse_port_direction,
    parse_property, port_bit_name, BitValue,
};

/// Find the top module in the JSON modules object.
///
/// The top module is the one with attribute `"top"` set to a value that
/// evaluates to 1. If no module has this attribute, the first (or only)
/// module is selected.
fn find_top_module<'a>(
    modules: &'a serde_json::Map<String, Value>,
) -> Result<(&'a str, &'a Value)> {
    if modules.is_empty() {
        bail!("No modules found in JSON");
    }

    // Look for a module with attribute top=1
    for (name, module) in modules {
        if let Some(attrs) = module.get("attributes").and_then(|a| a.as_object()) {
            if let Some(top_val) = attrs.get("top") {
                let prop = parse_property(top_val)?;
                if let Some(v) = prop.as_int() {
                    if v != 0 {
                        return Ok((name.as_str(), module));
                    }
                }
            }
        }
    }

    // No explicit top attribute found; use the first module.
    let (name, module) = modules.iter().next().unwrap();
    Ok((name.as_str(), module))
}

/// Parse a Yosys JSON netlist string and populate a [`Design`].
///
/// # Arguments
///
/// * `json_str` - The JSON string produced by `yosys -o design.json`
/// * `pool` - The string interning pool to use for all `IdString` values
///
/// # Returns
///
/// A fully populated [`Design`] with cells, nets, and hierarchy.
///
/// # Errors
///
/// Returns an error if the JSON is malformed, missing required fields, or
/// contains unsupported constructs.
pub fn parse_json(json_str: &str, pool: &IdStringPool) -> Result<Design> {
    let json: Value = serde_json::from_str(json_str).context("Failed to parse JSON")?;

    let modules = json
        .get("modules")
        .and_then(|m| m.as_object())
        .context("Missing or invalid 'modules' key in JSON")?;

    let (top_name, top_module) = find_top_module(modules)?;

    let mut design = Design::new();
    design.top_module = pool.intern(top_name);

    parse_module(top_module, &mut design, pool)
        .with_context(|| format!("Failed to parse module '{}'", top_name))?;

    Ok(design)
}

/// Parse a single Yosys JSON module into the design.
fn parse_module(module: &Value, design: &mut Design, pool: &IdStringPool) -> Result<()> {
    // Step 1: Scan all bit indices across cells and ports, create nets for each.
    let mut bit_to_net: FxHashMap<i64, NetId> = FxHashMap::default();
    collect_bit_indices(module, &mut bit_to_net, design, pool)?;

    // Step 2: Create constant driver nets and cells.
    let gnd_net = create_constant_net(design, pool, "$PACKER_GND_NET");
    let vcc_net = create_constant_net(design, pool, "$PACKER_VCC_NET");
    create_constant_driver(design, pool, "$PACKER_GND", "GND", "Y", gnd_net)?;
    create_constant_driver(design, pool, "$PACKER_VCC", "VCC", "Y", vcc_net)?;

    // Step 3: Parse cells.
    if let Some(cells) = module.get("cells").and_then(|c| c.as_object()) {
        for (cell_name, cell_json) in cells {
            parse_cell(
                cell_name,
                cell_json,
                design,
                pool,
                &bit_to_net,
                gnd_net,
                vcc_net,
            )
            .with_context(|| format!("Failed to parse cell '{}'", cell_name))?;
        }
    }

    // Step 4: Parse top-level ports.
    if let Some(ports) = module.get("ports").and_then(|p| p.as_object()) {
        for (port_name, port_json) in ports {
            parse_top_port(
                port_name,
                port_json,
                design,
                pool,
                &bit_to_net,
                gnd_net,
                vcc_net,
            )
            .with_context(|| format!("Failed to parse top-level port '{}'", port_name))?;
        }
    }

    // Step 5: Apply net names from the "netnames" section. Top-level port names
    // outrank every other label for a net, so collect them first.
    if let Some(netnames) = module.get("netnames").and_then(|n| n.as_object()) {
        let top_ports: rustc_hash::FxHashSet<String> = module
            .get("ports")
            .and_then(|p| p.as_object())
            .map(|ports| ports.keys().cloned().collect())
            .unwrap_or_default();
        apply_net_names(netnames, design, pool, &bit_to_net, &top_ports)?;
    }

    Ok(())
}

/// Parse a single cell from the JSON into the design.
fn parse_cell(
    cell_name: &str,
    cell_json: &Value,
    design: &mut Design,
    pool: &IdStringPool,
    bit_to_net: &FxHashMap<i64, NetId>,
    gnd_net: NetId,
    vcc_net: NetId,
) -> Result<()> {
    let cell_type_str = cell_json
        .get("type")
        .and_then(|t| t.as_str())
        .context("Cell missing 'type' field")?;

    let name_id = pool.intern(cell_name);
    let type_id = pool.intern(cell_type_str);

    let cell_idx = design.add_cell(name_id, type_id);

    // Parse port_directions to build a map of port name -> direction
    let port_dirs: FxHashMap<String, PortType> =
        if let Some(dirs) = cell_json.get("port_directions").and_then(|d| d.as_object()) {
            let mut map = FxHashMap::default();
            for (pname, dir_val) in dirs {
                let dir_str = dir_val
                    .as_str()
                    .context("port_direction value must be a string")?;
                map.insert(pname.clone(), parse_port_direction(dir_str)?);
            }
            map
        } else {
            FxHashMap::default()
        };

    // Parse connections and create ports
    if let Some(conns) = cell_json.get("connections").and_then(|c| c.as_object()) {
        for (port_name, bits_val) in conns {
            let bits = bits_val
                .as_array()
                .context("Connection bits must be an array")?;

            let port_type = port_dirs
                .get(port_name)
                .copied()
                .unwrap_or_else(|| infer_port_direction(cell_type_str, port_name));
            let total_bits = bits.len();

            for (i, bit) in bits.iter().enumerate() {
                let bit_val = parse_bit_value(bit)?;
                let actual_port_name = port_bit_name(port_name, i, total_bits);
                let port_id = pool.intern(&actual_port_name);

                // Add port to cell
                design.cell_edit(cell_idx).add_port(port_id, port_type);

                // Determine which net this bit connects to
                let net_idx = match &bit_val {
                    BitValue::Signal(idx) => Some(
                        *bit_to_net
                            .get(idx)
                            .context("Signal bit index not found in net map")?,
                    ),
                    BitValue::Zero => Some(gnd_net),
                    BitValue::One => Some(vcc_net),
                    BitValue::Undef => None,
                };

                if let Some(net_idx) = net_idx {
                    connect_port_to_net(design, cell_idx, port_id, port_type, net_idx)?;
                }
            }
        }
    }

    // Parse parameters
    if let Some(params) = cell_json.get("parameters").and_then(|p| p.as_object()) {
        for (param_name, param_val) in params {
            let key = pool.intern(param_name);
            let prop = parse_property(param_val)?;
            design.cell_edit(cell_idx).set_param(key, prop);
        }
    }

    // Parse attributes
    if let Some(attrs) = cell_json.get("attributes").and_then(|a| a.as_object()) {
        for (attr_name, attr_val) in attrs {
            let key = pool.intern(attr_name);
            let prop = parse_property(attr_val)?;
            design.cell_edit(cell_idx).set_attr(key, prop);
        }
    }

    Ok(())
}

/// Parse a top-level port and create a pseudo-cell for it.
///
/// Input ports get `$nextpnr_IBUF` pseudo-cells (output drives the internal net).
/// Output ports get `$nextpnr_OBUF` pseudo-cells (input reads from the internal net).
/// Inout ports get `$nextpnr_IOBUF` pseudo-cells (bidirectional).
fn parse_top_port(
    port_name: &str,
    port_json: &Value,
    design: &mut Design,
    pool: &IdStringPool,
    bit_to_net: &FxHashMap<i64, NetId>,
    gnd_net: NetId,
    vcc_net: NetId,
) -> Result<()> {
    let dir_str = port_json
        .get("direction")
        .and_then(|d| d.as_str())
        .context("Top-level port missing 'direction'")?;
    let port_dir = parse_port_direction(dir_str)?;

    let bits = port_json
        .get("bits")
        .and_then(|b| b.as_array())
        .context("Top-level port missing 'bits'")?;

    let total_bits = bits.len();

    for (i, bit) in bits.iter().enumerate() {
        let bit_val = parse_bit_value(bit)?;
        let actual_port_name = port_bit_name(port_name, i, total_bits);

        // Determine the pseudo-cell type and internal port name
        let (cell_type, internal_port_name, internal_port_type) = match port_dir {
            PortType::In => ("$nextpnr_IBUF", "O", PortType::Out),
            PortType::Out => ("$nextpnr_OBUF", "I", PortType::In),
            PortType::InOut => ("$nextpnr_IOBUF", "IO", PortType::InOut),
        };

        let cell_name = format!("$io${}", actual_port_name);
        let cell_name_id = pool.intern(&cell_name);
        let cell_type_id = pool.intern(cell_type);
        let internal_port_id = pool.intern(internal_port_name);

        let cell_idx = design.add_cell(cell_name_id, cell_type_id);

        // Add the internal port to the pseudo-cell
        design
            .cell_edit(cell_idx)
            .add_port(internal_port_id, internal_port_type);

        // Connect to the corresponding net
        let net_idx = match &bit_val {
            BitValue::Signal(idx) => Some(
                *bit_to_net
                    .get(idx)
                    .context("Signal bit index not found in net map")?,
            ),
            BitValue::Zero => Some(gnd_net),
            BitValue::One => Some(vcc_net),
            BitValue::Undef => None,
        };

        if let Some(net_idx) = net_idx {
            connect_port_to_net(
                design,
                cell_idx,
                internal_port_id,
                internal_port_type,
                net_idx,
            )?;
        }
    }

    Ok(())
}
