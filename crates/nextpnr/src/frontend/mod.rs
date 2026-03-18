//! Yosys JSON netlist parser for the nextpnr-rust FPGA place-and-route tool.
//!
//! This module reads the JSON output produced by Yosys (`write_json`) and
//! populates a [`Design`](crate::netlist::Design) with cells, nets, ports, parameters,
//! and attributes.

use crate::netlist::{CellId, Design, NetId};
use crate::common::{IdString, IdStringPool};
use crate::netlist::{PortType, Property};
use anyhow::{bail, Context, Result};
use rustc_hash::FxHashMap;
use serde_json::Value;

/// A bit value in a Yosys JSON connection or port bits array.
///
/// Yosys represents connection bits as either integers (net indices) or
/// string constants ("0", "1", "x"/"z").
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum BitValue {
    /// A signal net identified by a unique integer index.
    Signal(i64),
    /// Constant logic 0.
    Zero,
    /// Constant logic 1.
    One,
    /// Don't care / unconnected / high-impedance.
    Undef,
}

/// Parse a single element from a Yosys `bits` or `connections` array.
pub fn parse_bit_value(val: &Value) -> Result<BitValue> {
    match val {
        Value::Number(n) => {
            let idx = n.as_i64().context("Bit index is not a valid integer")?;
            Ok(BitValue::Signal(idx))
        }
        Value::String(s) => match s.as_str() {
            "0" => Ok(BitValue::Zero),
            "1" => Ok(BitValue::One),
            "x" | "z" => Ok(BitValue::Undef),
            other => bail!("Unknown constant bit value: {:?}", other),
        },
        _ => bail!("Invalid bit value in connections array: {:?}", val),
    }
}

/// Infer port direction from cell type and port name when `port_directions`
/// is missing from the Yosys JSON (common with BLIF import).
fn infer_port_direction(cell_type: &str, port_name: &str) -> PortType {
    match cell_type {
        "LUT4" => {
            if port_name == "F" { PortType::Out } else { PortType::In }
        }
        "CARRY4" => {
            if port_name.starts_with("CO") || port_name.starts_with("O") {
                PortType::Out
            } else {
                PortType::In
            }
        }
        "GND" | "VCC" | "GND_DRV" | "VCC_DRV" => PortType::Out,
        "IOB" => PortType::InOut,
        _ => {
            // Heuristic: common output port names across LUT, DFF, BUF, IBUF, OBUF, etc.
            if port_name == "Q" || port_name == "O" || port_name == "F" || port_name == "Y" {
                PortType::Out
            } else {
                PortType::In
            }
        }
    }
}

/// Parse a Yosys port direction string into a [`PortType`].
pub fn parse_port_direction(dir: &str) -> Result<PortType> {
    match dir {
        "input" => Ok(PortType::In),
        "output" => Ok(PortType::Out),
        "inout" => Ok(PortType::InOut),
        other => bail!("Unknown port direction: {:?}", other),
    }
}

/// Parse a Yosys property value (parameter or attribute) into a [`Property`].
///
/// Yosys represents parameters as:
///  - Binary strings like `"0000000000001111"` for LUT INIT values
///  - Decimal integer strings
///  - Arbitrary strings (attributes like `"src": "blinky.v:5"`)
///  - Integer JSON values
///
/// We try to determine if a string is a binary bit-vector (only '0'/'1' chars
/// and either long or explicitly bit-patterned) or a plain string/number.
pub fn parse_property(val: &Value) -> Result<Property> {
    match val {
        Value::Number(n) => {
            let v = n.as_i64().context("Property number is not a valid i64")?;
            Ok(Property::int(v))
        }
        Value::String(s) => {
            // Yosys encodes parameters as binary strings of '0' and '1'.
            // Attributes can be binary-encoded integers (32+ chars of 0/1)
            // or plain strings.
            if !s.is_empty() && s.chars().all(|c| c == '0' || c == '1') {
                Ok(Property::bit_vector(s.clone()))
            } else {
                Ok(Property::string(s.clone()))
            }
        }
        _ => bail!("Unsupported property value type: {:?}", val),
    }
}

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

/// Compute the port name for a potentially multi-bit port.
///
/// If a port has a single bit, its name is used as-is.
/// If a port has multiple bits, each bit gets an indexed name: `port[0]`, `port[1]`, etc.
pub fn port_bit_name(base_name: &str, bit_index: usize, total_bits: usize) -> String {
    if total_bits == 1 {
        base_name.to_string()
    } else {
        format!("{}[{}]", base_name, bit_index)
    }
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

    // Step 5: Apply net names from the "netnames" section.
    if let Some(netnames) = module.get("netnames").and_then(|n| n.as_object()) {
        apply_net_names(netnames, design, pool, &bit_to_net)?;
    }

    Ok(())
}

/// Scan the module JSON to collect all unique signal bit indices and create a
/// net for each one. Returns a mapping from bit index to `NetIdx`.
fn collect_bit_indices(
    module: &Value,
    bit_to_net: &mut FxHashMap<i64, NetId>,
    design: &mut Design,
    pool: &IdStringPool,
) -> Result<()> {
    // Collect from cell connections
    if let Some(cells) = module.get("cells").and_then(|c| c.as_object()) {
        for (_cell_name, cell_json) in cells {
            if let Some(conns) = cell_json.get("connections").and_then(|c| c.as_object()) {
                for (_port_name, bits_val) in conns {
                    let bits = bits_val
                        .as_array()
                        .context("Cell connection bits must be an array")?;
                    for bit in bits {
                        if let BitValue::Signal(idx) = parse_bit_value(bit)? {
                            if !bit_to_net.contains_key(&idx) {
                                let net_name = format!("$signal${}", idx);
                                let net_idx = design.add_net(pool.intern(&net_name));
                                bit_to_net.insert(idx, net_idx);
                            }
                        }
                    }
                }
            }
        }
    }

    // Collect from top-level port bits
    if let Some(ports) = module.get("ports").and_then(|p| p.as_object()) {
        for (_port_name, port_json) in ports {
            if let Some(bits) = port_json.get("bits").and_then(|b| b.as_array()) {
                for bit in bits {
                    if let BitValue::Signal(idx) = parse_bit_value(bit)? {
                        if !bit_to_net.contains_key(&idx) {
                            let net_name = format!("$signal${}", idx);
                            let net_idx = design.add_net(pool.intern(&net_name));
                            bit_to_net.insert(idx, net_idx);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

/// Create a constant net (GND or VCC).
fn create_constant_net(design: &mut Design, pool: &IdStringPool, name: &str) -> NetId {
    design.add_net(pool.intern(name))
}

/// Create a constant driver cell (GND or VCC) and connect its output to the
/// given net.
fn create_constant_driver(
    design: &mut Design,
    pool: &IdStringPool,
    cell_name: &str,
    cell_type: &str,
    output_port: &str,
    net_idx: NetId,
) -> Result<()> {
    let name_id = pool.intern(cell_name);
    let type_id = pool.intern(cell_type);
    let port_id = pool.intern(output_port);

    let cell_idx = design.add_cell(name_id, type_id);

    // Add the output port
    design.cell_edit(cell_idx).add_port(port_id, PortType::Out);

    // Connect: set the port's net and set the net's driver
    design
        .cell_edit(cell_idx)
        .set_port_net(port_id, Some(net_idx), None);
    design.net_edit(net_idx).set_driver(cell_idx, port_id);

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

/// Connect a port on a cell to a net, updating both the port and the net
/// (driver or user reference).
fn connect_port_to_net(
    design: &mut Design,
    cell_idx: CellId,
    port_id: IdString,
    port_type: PortType,
    net_idx: NetId,
) -> Result<()> {
    match port_type {
        PortType::Out => {
            // Output port: this port drives the net.
            design
                .cell_edit(cell_idx)
                .set_port_net(port_id, Some(net_idx), None);
            design.net_edit(net_idx).set_driver(cell_idx, port_id);
        }
        PortType::In | PortType::InOut => {
            // Input or bidirectional port: this port is a user of the net.
            let user_idx = design.net_edit(net_idx).add_user(cell_idx, port_id);
            design
                .cell_edit(cell_idx)
                .set_port_net(port_id, Some(net_idx), Some(user_idx));
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

/// Apply human-readable names from the `netnames` section to nets.
///
/// Yosys includes a `netnames` section that maps names to bit indices. We use
/// this to rename nets from their synthetic `$signal$N` names to the actual
/// signal names from the HDL source.
fn apply_net_names(
    netnames: &serde_json::Map<String, Value>,
    design: &mut Design,
    pool: &IdStringPool,
    bit_to_net: &FxHashMap<i64, NetId>,
) -> Result<()> {
    for (net_name, nn_json) in netnames {
        let hide_name = nn_json
            .get("hide_name")
            .and_then(|h| h.as_i64())
            .unwrap_or(0);

        // Only apply names that are not hidden (hide_name == 0)
        if hide_name != 0 {
            continue;
        }

        let bits = nn_json
            .get("bits")
            .and_then(|b| b.as_array())
            .context("netnames entry missing 'bits'")?;

        let total_bits = bits.len();

        for (i, bit) in bits.iter().enumerate() {
            if let BitValue::Signal(idx) = parse_bit_value(bit)? {
                if let Some(&net_idx) = bit_to_net.get(&idx) {
                    let actual_name = port_bit_name(net_name, i, total_bits);
                    let name_id = pool.intern(&actual_name);

                    design.rename_net(net_idx, name_id);
                }
            }
        }

        // Apply attributes to the net
        if let Some(attrs) = nn_json.get("attributes").and_then(|a| a.as_object()) {
            // Apply to the first signal bit's net (if it exists)
            if let Some(first_bit) = bits.first() {
                if let BitValue::Signal(idx) = parse_bit_value(first_bit)? {
                    if let Some(&net_idx) = bit_to_net.get(&idx) {
                        for (attr_name, attr_val) in attrs {
                            let key = pool.intern(attr_name);
                            let prop = parse_property(attr_val)?;
                            design.net_edit(net_idx).set_attr(key, prop);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}
