//! Yosys JSON netlist parser for the nextpnr-rust FPGA place-and-route tool.
//!
//! This crate reads the JSON output produced by Yosys (`write_json`) and
//! populates an [`npnr_netlist::Design`] with cells, nets, ports, parameters,
//! and attributes.

use anyhow::{bail, Context, Result};
use npnr_netlist::{CellIdx, Design, NetIdx, PortRef};
use npnr_types::{IdString, IdStringPool, PortType, Property};
use rustc_hash::FxHashMap;
use serde_json::Value;

/// A bit value in a Yosys JSON connection or port bits array.
///
/// Yosys represents connection bits as either integers (net indices) or
/// string constants ("0", "1", "x"/"z").
#[derive(Debug, Clone, PartialEq, Eq)]
enum BitValue {
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
fn parse_bit_value(val: &Value) -> Result<BitValue> {
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

/// Parse a Yosys port direction string into a [`PortType`].
fn parse_port_direction(dir: &str) -> Result<PortType> {
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
fn parse_property(val: &Value) -> Result<Property> {
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
fn port_bit_name(base_name: &str, bit_index: usize, total_bits: usize) -> String {
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
    let mut bit_to_net: FxHashMap<i64, NetIdx> = FxHashMap::default();
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
                cell_name, cell_json, design, pool, &bit_to_net, gnd_net, vcc_net,
            )
            .with_context(|| format!("Failed to parse cell '{}'", cell_name))?;
        }
    }

    // Step 4: Parse top-level ports.
    if let Some(ports) = module.get("ports").and_then(|p| p.as_object()) {
        for (port_name, port_json) in ports {
            parse_top_port(
                port_name, port_json, design, pool, &bit_to_net, gnd_net, vcc_net,
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
    bit_to_net: &mut FxHashMap<i64, NetIdx>,
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
fn create_constant_net(design: &mut Design, pool: &IdStringPool, name: &str) -> NetIdx {
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
    net_idx: NetIdx,
) -> Result<()> {
    let name_id = pool.intern(cell_name);
    let type_id = pool.intern(cell_type);
    let port_id = pool.intern(output_port);

    let cell_idx = design.add_cell(name_id, type_id);

    // Add the output port
    let cell = design.cell_mut(cell_idx);
    cell.add_port(port_id, PortType::Out);

    // Connect: set the port's net and set the net's driver
    let cell = design.cell_mut(cell_idx);
    let port = cell
        .port_mut(port_id)
        .context("Port not found on constant driver cell")?;
    port.net = net_idx;
    port.user_idx = -1; // driver, not a user

    let net = design.net_mut(net_idx);
    net.driver = PortRef {
        cell: cell_idx,
        port: port_id,
        budget: 0,
    };

    Ok(())
}

/// Parse a single cell from the JSON into the design.
fn parse_cell(
    cell_name: &str,
    cell_json: &Value,
    design: &mut Design,
    pool: &IdStringPool,
    bit_to_net: &FxHashMap<i64, NetIdx>,
    gnd_net: NetIdx,
    vcc_net: NetIdx,
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

            let port_type = port_dirs.get(port_name).copied().unwrap_or(PortType::In);
            let total_bits = bits.len();

            for (i, bit) in bits.iter().enumerate() {
                let bit_val = parse_bit_value(bit)?;
                let actual_port_name = port_bit_name(port_name, i, total_bits);
                let port_id = pool.intern(&actual_port_name);

                // Add port to cell
                let cell = design.cell_mut(cell_idx);
                cell.add_port(port_id, port_type);

                // Determine which net this bit connects to
                let net_idx = match &bit_val {
                    BitValue::Signal(idx) => {
                        Some(*bit_to_net.get(idx).context("Signal bit index not found in net map")?)
                    }
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
            design.cell_mut(cell_idx).params.insert(key, prop);
        }
    }

    // Parse attributes
    if let Some(attrs) = cell_json.get("attributes").and_then(|a| a.as_object()) {
        for (attr_name, attr_val) in attrs {
            let key = pool.intern(attr_name);
            let prop = parse_property(attr_val)?;
            design.cell_mut(cell_idx).attrs.insert(key, prop);
        }
    }

    Ok(())
}

/// Connect a port on a cell to a net, updating both the port and the net
/// (driver or user reference).
fn connect_port_to_net(
    design: &mut Design,
    cell_idx: CellIdx,
    port_id: IdString,
    port_type: PortType,
    net_idx: NetIdx,
) -> Result<()> {
    match port_type {
        PortType::Out => {
            // Output port: this port drives the net.
            let cell = design.cell_mut(cell_idx);
            let port = cell
                .port_mut(port_id)
                .context("Port not found on cell")?;
            port.net = net_idx;
            port.user_idx = -1;

            let net = design.net_mut(net_idx);
            net.driver = PortRef {
                cell: cell_idx,
                port: port_id,
                budget: 0,
            };
        }
        PortType::In | PortType::InOut => {
            // Input or bidirectional port: this port is a user of the net.
            let net = design.net_mut(net_idx);
            let user_idx = net.users.len() as i32;
            net.users.push(PortRef {
                cell: cell_idx,
                port: port_id,
                budget: 0,
            });

            let cell = design.cell_mut(cell_idx);
            let port = cell
                .port_mut(port_id)
                .context("Port not found on cell")?;
            port.net = net_idx;
            port.user_idx = user_idx;
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
    bit_to_net: &FxHashMap<i64, NetIdx>,
    gnd_net: NetIdx,
    vcc_net: NetIdx,
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
        let cell = design.cell_mut(cell_idx);
        cell.add_port(internal_port_id, internal_port_type);

        // Connect to the corresponding net
        let net_idx = match &bit_val {
            BitValue::Signal(idx) => {
                Some(*bit_to_net.get(idx).context("Signal bit index not found in net map")?)
            }
            BitValue::Zero => Some(gnd_net),
            BitValue::One => Some(vcc_net),
            BitValue::Undef => None,
        };

        if let Some(net_idx) = net_idx {
            connect_port_to_net(design, cell_idx, internal_port_id, internal_port_type, net_idx)?;
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
    bit_to_net: &FxHashMap<i64, NetIdx>,
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

                    // Update the net's name
                    let net = design.net_mut(net_idx);
                    net.name = name_id;

                    // Update the name lookup map: remove old name, insert new
                    let old_name = {
                        let mut old = IdString::EMPTY;
                        for (&k, &v) in design.nets.iter() {
                            if v == net_idx {
                                old = k;
                                break;
                            }
                        }
                        old
                    };
                    if !old_name.is_empty() {
                        design.nets.remove(&old_name);
                    }
                    design.nets.insert(name_id, net_idx);
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
                            design.net_mut(net_idx).attrs.insert(key, prop);
                        }
                    }
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_pool() -> IdStringPool {
        IdStringPool::new()
    }

    // =========================================================================
    // Helper unit tests
    // =========================================================================

    #[test]
    fn test_parse_bit_value_signal() {
        let val = serde_json::json!(42);
        assert_eq!(parse_bit_value(&val).unwrap(), BitValue::Signal(42));
    }

    #[test]
    fn test_parse_bit_value_zero() {
        let val = serde_json::json!("0");
        assert_eq!(parse_bit_value(&val).unwrap(), BitValue::Zero);
    }

    #[test]
    fn test_parse_bit_value_one() {
        let val = serde_json::json!("1");
        assert_eq!(parse_bit_value(&val).unwrap(), BitValue::One);
    }

    #[test]
    fn test_parse_bit_value_x() {
        let val = serde_json::json!("x");
        assert_eq!(parse_bit_value(&val).unwrap(), BitValue::Undef);
    }

    #[test]
    fn test_parse_bit_value_z() {
        let val = serde_json::json!("z");
        assert_eq!(parse_bit_value(&val).unwrap(), BitValue::Undef);
    }

    #[test]
    fn test_parse_bit_value_invalid_string() {
        let val = serde_json::json!("abc");
        assert!(parse_bit_value(&val).is_err());
    }

    #[test]
    fn test_parse_bit_value_invalid_type() {
        let val = serde_json::json!(true);
        assert!(parse_bit_value(&val).is_err());
    }

    #[test]
    fn test_parse_port_direction() {
        assert_eq!(parse_port_direction("input").unwrap(), PortType::In);
        assert_eq!(parse_port_direction("output").unwrap(), PortType::Out);
        assert_eq!(parse_port_direction("inout").unwrap(), PortType::InOut);
        assert!(parse_port_direction("unknown").is_err());
    }

    #[test]
    fn test_parse_property_integer() {
        let val = serde_json::json!(42);
        let prop = parse_property(&val).unwrap();
        assert_eq!(prop, Property::int(42));
    }

    #[test]
    fn test_parse_property_binary_string() {
        let val = serde_json::json!("0000000000001111");
        let prop = parse_property(&val).unwrap();
        assert_eq!(prop, Property::bit_vector("0000000000001111"));
        assert_eq!(prop.as_int(), Some(15));
    }

    #[test]
    fn test_parse_property_plain_string() {
        let val = serde_json::json!("blinky.v:5.3-5.20");
        let prop = parse_property(&val).unwrap();
        assert_eq!(prop, Property::string("blinky.v:5.3-5.20"));
    }

    #[test]
    fn test_port_bit_name_single() {
        assert_eq!(port_bit_name("clk", 0, 1), "clk");
    }

    #[test]
    fn test_port_bit_name_multi() {
        assert_eq!(port_bit_name("data", 0, 4), "data[0]");
        assert_eq!(port_bit_name("data", 3, 4), "data[3]");
    }

    // =========================================================================
    // Integration tests: minimal blinky design
    // =========================================================================

    const BLINKY_JSON: &str = r#"{
        "creator": "Yosys 0.9+4081",
        "modules": {
            "top": {
                "attributes": {
                    "top": "00000000000000000000000000000001"
                },
                "parameter_default_values": {},
                "ports": {
                    "clk": { "direction": "input", "bits": [2] },
                    "led": { "direction": "output", "bits": [3] }
                },
                "cells": {
                    "lut0": {
                        "hide_name": 0,
                        "type": "LUT4",
                        "parameters": {
                            "INIT": "0000000000001111"
                        },
                        "attributes": {
                            "src": "blinky.v:5.3-5.20"
                        },
                        "port_directions": {
                            "A": "input",
                            "B": "input",
                            "C": "input",
                            "D": "input",
                            "Z": "output"
                        },
                        "connections": {
                            "A": [2],
                            "B": ["0"],
                            "C": ["0"],
                            "D": ["0"],
                            "Z": [3]
                        }
                    }
                },
                "netnames": {
                    "clk": {
                        "hide_name": 0,
                        "bits": [2],
                        "attributes": {}
                    },
                    "led": {
                        "hide_name": 0,
                        "bits": [3],
                        "attributes": {}
                    }
                }
            }
        }
    }"#;

    #[test]
    fn test_blinky_parse_succeeds() {
        let pool = make_pool();
        let design = parse_json(BLINKY_JSON, &pool).unwrap();
        assert!(!design.top_module.is_empty());
        assert_eq!(pool.lookup(design.top_module).as_deref(), Some("top"));
    }

    #[test]
    fn test_blinky_has_cells() {
        let pool = make_pool();
        let design = parse_json(BLINKY_JSON, &pool).unwrap();

        // Should have: lut0, $PACKER_GND, $PACKER_VCC, $io$clk, $io$led
        let lut0_id = pool.intern("lut0");
        assert!(design.cell_by_name(lut0_id).is_some());

        let gnd_id = pool.intern("$PACKER_GND");
        assert!(design.cell_by_name(gnd_id).is_some());

        let vcc_id = pool.intern("$PACKER_VCC");
        assert!(design.cell_by_name(vcc_id).is_some());

        let io_clk_id = pool.intern("$io$clk");
        assert!(design.cell_by_name(io_clk_id).is_some());

        let io_led_id = pool.intern("$io$led");
        assert!(design.cell_by_name(io_led_id).is_some());
    }

    #[test]
    fn test_blinky_lut_type() {
        let pool = make_pool();
        let design = parse_json(BLINKY_JSON, &pool).unwrap();

        let lut0_id = pool.intern("lut0");
        let lut4_type = pool.intern("LUT4");
        let cell_idx = design.cell_by_name(lut0_id).unwrap();
        let cell = design.cell(cell_idx);
        assert_eq!(cell.cell_type, lut4_type);
    }

    #[test]
    fn test_blinky_lut_ports() {
        let pool = make_pool();
        let design = parse_json(BLINKY_JSON, &pool).unwrap();

        let lut0_id = pool.intern("lut0");
        let cell_idx = design.cell_by_name(lut0_id).unwrap();
        let cell = design.cell(cell_idx);

        // Check port A exists and is input
        let port_a = pool.intern("A");
        let pa = cell.port(port_a).unwrap();
        assert_eq!(pa.port_type, PortType::In);
        assert!(pa.net.is_some()); // connected to signal net 2

        // Check port Z exists and is output
        let port_z = pool.intern("Z");
        let pz = cell.port(port_z).unwrap();
        assert_eq!(pz.port_type, PortType::Out);
        assert!(pz.net.is_some()); // connected to signal net 3
    }

    #[test]
    fn test_blinky_lut_parameters() {
        let pool = make_pool();
        let design = parse_json(BLINKY_JSON, &pool).unwrap();

        let lut0_id = pool.intern("lut0");
        let cell_idx = design.cell_by_name(lut0_id).unwrap();
        let cell = design.cell(cell_idx);

        let init_key = pool.intern("INIT");
        let init = cell.params.get(&init_key).unwrap();
        assert_eq!(init.as_int(), Some(15)); // 0000000000001111 in binary = 15
    }

    #[test]
    fn test_blinky_lut_attributes() {
        let pool = make_pool();
        let design = parse_json(BLINKY_JSON, &pool).unwrap();

        let lut0_id = pool.intern("lut0");
        let cell_idx = design.cell_by_name(lut0_id).unwrap();
        let cell = design.cell(cell_idx);

        let src_key = pool.intern("src");
        let src = cell.attrs.get(&src_key).unwrap();
        assert_eq!(src.as_str(), "blinky.v:5.3-5.20");
    }

    #[test]
    fn test_blinky_constant_connections() {
        let pool = make_pool();
        let design = parse_json(BLINKY_JSON, &pool).unwrap();

        // Ports B, C, D of lut0 are connected to "0" -> GND net
        let lut0_id = pool.intern("lut0");
        let cell_idx = design.cell_by_name(lut0_id).unwrap();
        let cell = design.cell(cell_idx);

        let gnd_net_id = pool.intern("$PACKER_GND_NET");
        let gnd_net_idx = design.net_by_name(gnd_net_id).unwrap();

        let port_b = pool.intern("B");
        let pb = cell.port(port_b).unwrap();
        assert_eq!(pb.net, gnd_net_idx);

        let port_c = pool.intern("C");
        let pc = cell.port(port_c).unwrap();
        assert_eq!(pc.net, gnd_net_idx);

        let port_d = pool.intern("D");
        let pd = cell.port(port_d).unwrap();
        assert_eq!(pd.net, gnd_net_idx);
    }

    #[test]
    fn test_blinky_has_nets() {
        let pool = make_pool();
        let design = parse_json(BLINKY_JSON, &pool).unwrap();

        // Should have nets: clk (renamed from $signal$2), led (renamed from $signal$3),
        // $PACKER_GND_NET, $PACKER_VCC_NET
        let gnd_net_id = pool.intern("$PACKER_GND_NET");
        assert!(design.net_by_name(gnd_net_id).is_some());

        let vcc_net_id = pool.intern("$PACKER_VCC_NET");
        assert!(design.net_by_name(vcc_net_id).is_some());

        // The signal nets get renamed by netnames
        let clk_net = pool.intern("clk");
        assert!(design.net_by_name(clk_net).is_some());

        let led_net = pool.intern("led");
        assert!(design.net_by_name(led_net).is_some());
    }

    #[test]
    fn test_blinky_io_cells() {
        let pool = make_pool();
        let design = parse_json(BLINKY_JSON, &pool).unwrap();

        // clk is an input -> $nextpnr_IBUF pseudo-cell
        let io_clk_id = pool.intern("$io$clk");
        let ibuf_type = pool.intern("$nextpnr_IBUF");
        let clk_cell_idx = design.cell_by_name(io_clk_id).unwrap();
        let clk_cell = design.cell(clk_cell_idx);
        assert_eq!(clk_cell.cell_type, ibuf_type);

        // led is an output -> $nextpnr_OBUF pseudo-cell
        let io_led_id = pool.intern("$io$led");
        let obuf_type = pool.intern("$nextpnr_OBUF");
        let led_cell_idx = design.cell_by_name(io_led_id).unwrap();
        let led_cell = design.cell(led_cell_idx);
        assert_eq!(led_cell.cell_type, obuf_type);
    }

    #[test]
    fn test_blinky_net_connectivity() {
        let pool = make_pool();
        let design = parse_json(BLINKY_JSON, &pool).unwrap();

        // The clk net (signal 2) should be:
        //  - driven by the IBUF (output port O)
        //  - used by lut0 port A
        let clk_net_id = pool.intern("clk");
        let clk_net_idx = design.net_by_name(clk_net_id).unwrap();
        let clk_net = design.net(clk_net_idx);

        // The IBUF's O port drives this net
        assert!(clk_net.has_driver());
        let io_clk_id = pool.intern("$io$clk");
        let io_clk_idx = design.cell_by_name(io_clk_id).unwrap();
        assert_eq!(clk_net.driver.cell, io_clk_idx);

        // lut0's A port is a user
        assert!(clk_net.num_users() >= 1);
        let lut0_id = pool.intern("lut0");
        let lut0_idx = design.cell_by_name(lut0_id).unwrap();
        let has_lut_user = clk_net
            .users
            .iter()
            .any(|u| u.cell == lut0_idx && u.port == pool.intern("A"));
        assert!(has_lut_user);
    }

    // =========================================================================
    // Multi-bit port test
    // =========================================================================

    const MULTIBIT_JSON: &str = r#"{
        "creator": "Yosys",
        "modules": {
            "multi": {
                "attributes": { "top": "00000000000000000000000000000001" },
                "parameter_default_values": {},
                "ports": {
                    "data_in": { "direction": "input", "bits": [10, 11, 12, 13] },
                    "data_out": { "direction": "output", "bits": [20, 21, 22, 23] }
                },
                "cells": {
                    "buf0": {
                        "hide_name": 0,
                        "type": "BUF4",
                        "parameters": {},
                        "attributes": {},
                        "port_directions": {
                            "I": "input",
                            "O": "output"
                        },
                        "connections": {
                            "I": [10, 11, 12, 13],
                            "O": [20, 21, 22, 23]
                        }
                    }
                },
                "netnames": {}
            }
        }
    }"#;

    #[test]
    fn test_multibit_port_names() {
        let pool = make_pool();
        let design = parse_json(MULTIBIT_JSON, &pool).unwrap();

        let buf0_id = pool.intern("buf0");
        let cell_idx = design.cell_by_name(buf0_id).unwrap();
        let cell = design.cell(cell_idx);

        // Multi-bit port I should become I[0], I[1], I[2], I[3]
        for i in 0..4 {
            let pname = pool.intern(&format!("I[{}]", i));
            assert!(cell.port(pname).is_some(), "Missing port I[{}]", i);
            assert_eq!(cell.port(pname).unwrap().port_type, PortType::In);
        }

        // Multi-bit port O should become O[0], O[1], O[2], O[3]
        for i in 0..4 {
            let pname = pool.intern(&format!("O[{}]", i));
            assert!(cell.port(pname).is_some(), "Missing port O[{}]", i);
            assert_eq!(cell.port(pname).unwrap().port_type, PortType::Out);
        }
    }

    #[test]
    fn test_multibit_io_cells() {
        let pool = make_pool();
        let design = parse_json(MULTIBIT_JSON, &pool).unwrap();

        // Multi-bit input port data_in -> 4 IBUF cells
        for i in 0..4 {
            let cell_name = pool.intern(&format!("$io$data_in[{}]", i));
            assert!(
                design.cell_by_name(cell_name).is_some(),
                "Missing IO cell for data_in[{}]",
                i
            );
            let cell_idx = design.cell_by_name(cell_name).unwrap();
            let cell = design.cell(cell_idx);
            assert_eq!(cell.cell_type, pool.intern("$nextpnr_IBUF"));
        }

        // Multi-bit output port data_out -> 4 OBUF cells
        for i in 0..4 {
            let cell_name = pool.intern(&format!("$io$data_out[{}]", i));
            assert!(
                design.cell_by_name(cell_name).is_some(),
                "Missing IO cell for data_out[{}]",
                i
            );
            let cell_idx = design.cell_by_name(cell_name).unwrap();
            let cell = design.cell(cell_idx);
            assert_eq!(cell.cell_type, pool.intern("$nextpnr_OBUF"));
        }
    }

    #[test]
    fn test_multibit_net_connections() {
        let pool = make_pool();
        let design = parse_json(MULTIBIT_JSON, &pool).unwrap();

        let buf0_id = pool.intern("buf0");
        let buf0_idx = design.cell_by_name(buf0_id).unwrap();

        // Each input bit of buf0 should be connected to the corresponding IBUF output
        for i in 0..4 {
            let port_name = pool.intern(&format!("I[{}]", i));
            let port = design.cell(buf0_idx).port(port_name).unwrap();
            assert!(port.net.is_some(), "buf0 I[{}] should be connected", i);
        }

        // Each output bit of buf0 should drive a net
        for i in 0..4 {
            let port_name = pool.intern(&format!("O[{}]", i));
            let port = design.cell(buf0_idx).port(port_name).unwrap();
            assert!(port.net.is_some(), "buf0 O[{}] should be connected", i);
        }
    }

    // =========================================================================
    // Constant driver bits test
    // =========================================================================

    const CONSTANTS_JSON: &str = r#"{
        "creator": "Yosys",
        "modules": {
            "consts": {
                "attributes": { "top": "00000000000000000000000000000001" },
                "parameter_default_values": {},
                "ports": {
                    "out_a": { "direction": "output", "bits": [5] },
                    "out_b": { "direction": "output", "bits": [6] }
                },
                "cells": {
                    "cell_a": {
                        "hide_name": 0,
                        "type": "BUF",
                        "parameters": {},
                        "attributes": {},
                        "port_directions": { "I": "input", "O": "output" },
                        "connections": { "I": ["1"], "O": [5] }
                    },
                    "cell_b": {
                        "hide_name": 0,
                        "type": "BUF",
                        "parameters": {},
                        "attributes": {},
                        "port_directions": { "I": "input", "O": "output" },
                        "connections": { "I": ["x"], "O": [6] }
                    }
                },
                "netnames": {}
            }
        }
    }"#;

    #[test]
    fn test_constant_one_connection() {
        let pool = make_pool();
        let design = parse_json(CONSTANTS_JSON, &pool).unwrap();

        let cell_a_id = pool.intern("cell_a");
        let cell_idx = design.cell_by_name(cell_a_id).unwrap();
        let cell = design.cell(cell_idx);

        let port_i = pool.intern("I");
        let pi = cell.port(port_i).unwrap();

        // Should be connected to VCC net
        let vcc_net_id = pool.intern("$PACKER_VCC_NET");
        let vcc_net_idx = design.net_by_name(vcc_net_id).unwrap();
        assert_eq!(pi.net, vcc_net_idx);
    }

    #[test]
    fn test_constant_x_unconnected() {
        let pool = make_pool();
        let design = parse_json(CONSTANTS_JSON, &pool).unwrap();

        let cell_b_id = pool.intern("cell_b");
        let cell_idx = design.cell_by_name(cell_b_id).unwrap();
        let cell = design.cell(cell_idx);

        // "x" means unconnected
        let port_i = pool.intern("I");
        let pi = cell.port(port_i).unwrap();
        assert!(pi.net.is_none());
    }

    // =========================================================================
    // Empty module test
    // =========================================================================

    #[test]
    fn test_empty_module() {
        let json = r#"{
            "creator": "Yosys",
            "modules": {
                "empty": {
                    "attributes": {},
                    "parameter_default_values": {},
                    "ports": {},
                    "cells": {},
                    "netnames": {}
                }
            }
        }"#;
        let pool = make_pool();
        let design = parse_json(json, &pool).unwrap();

        // Should have only the GND/VCC cells and nets
        assert_eq!(design.cell_store.len(), 2); // GND + VCC
        assert_eq!(design.net_store.len(), 2); // GND_NET + VCC_NET
    }

    // =========================================================================
    // No modules error test
    // =========================================================================

    #[test]
    fn test_no_modules_error() {
        let json = r#"{ "creator": "Yosys" }"#;
        let pool = make_pool();
        assert!(parse_json(json, &pool).is_err());
    }

    #[test]
    fn test_empty_modules_error() {
        let json = r#"{ "modules": {} }"#;
        let pool = make_pool();
        assert!(parse_json(json, &pool).is_err());
    }

    #[test]
    fn test_invalid_json_error() {
        let pool = make_pool();
        assert!(parse_json("not json at all", &pool).is_err());
    }

    // =========================================================================
    // Top module selection test
    // =========================================================================

    #[test]
    fn test_top_module_selection_by_attribute() {
        let json = r#"{
            "modules": {
                "sub": {
                    "attributes": {},
                    "parameter_default_values": {},
                    "ports": {},
                    "cells": {},
                    "netnames": {}
                },
                "main": {
                    "attributes": { "top": "00000000000000000000000000000001" },
                    "parameter_default_values": {},
                    "ports": {},
                    "cells": {},
                    "netnames": {}
                }
            }
        }"#;
        let pool = make_pool();
        let design = parse_json(json, &pool).unwrap();
        assert_eq!(pool.lookup(design.top_module).as_deref(), Some("main"));
    }

    // =========================================================================
    // Inout port test
    // =========================================================================

    #[test]
    fn test_inout_port() {
        let json = r#"{
            "modules": {
                "bidir": {
                    "attributes": { "top": "00000000000000000000000000000001" },
                    "parameter_default_values": {},
                    "ports": {
                        "sda": { "direction": "inout", "bits": [2] }
                    },
                    "cells": {},
                    "netnames": {}
                }
            }
        }"#;
        let pool = make_pool();
        let design = parse_json(json, &pool).unwrap();

        let io_sda = pool.intern("$io$sda");
        let cell_idx = design.cell_by_name(io_sda).unwrap();
        let cell = design.cell(cell_idx);
        assert_eq!(cell.cell_type, pool.intern("$nextpnr_IOBUF"));
    }

    // =========================================================================
    // GND net users test
    // =========================================================================

    #[test]
    fn test_gnd_net_has_users() {
        let pool = make_pool();
        let design = parse_json(BLINKY_JSON, &pool).unwrap();

        let gnd_net_id = pool.intern("$PACKER_GND_NET");
        let gnd_net_idx = design.net_by_name(gnd_net_id).unwrap();
        let gnd_net = design.net(gnd_net_idx);

        // GND net should have users: lut0 ports B, C, D
        assert_eq!(gnd_net.num_users(), 3);
    }

    #[test]
    fn test_gnd_net_has_driver() {
        let pool = make_pool();
        let design = parse_json(BLINKY_JSON, &pool).unwrap();

        let gnd_net_id = pool.intern("$PACKER_GND_NET");
        let gnd_net_idx = design.net_by_name(gnd_net_id).unwrap();
        let gnd_net = design.net(gnd_net_idx);

        assert!(gnd_net.has_driver());
        let gnd_cell_id = pool.intern("$PACKER_GND");
        let gnd_cell_idx = design.cell_by_name(gnd_cell_id).unwrap();
        assert_eq!(gnd_net.driver.cell, gnd_cell_idx);
    }

    // =========================================================================
    // Multiple parameters test
    // =========================================================================

    #[test]
    fn test_multiple_parameters() {
        let json = r#"{
            "modules": {
                "top": {
                    "attributes": { "top": "00000000000000000000000000000001" },
                    "parameter_default_values": {},
                    "ports": {},
                    "cells": {
                        "dff0": {
                            "hide_name": 0,
                            "type": "FDRE",
                            "parameters": {
                                "INIT": "0",
                                "IS_C_INVERTED": "1",
                                "SLICE": "SLICE_X0Y0"
                            },
                            "attributes": {
                                "src": "design.v:10"
                            },
                            "port_directions": {},
                            "connections": {}
                        }
                    },
                    "netnames": {}
                }
            }
        }"#;
        let pool = make_pool();
        let design = parse_json(json, &pool).unwrap();

        let cell_id = pool.intern("dff0");
        let cell_idx = design.cell_by_name(cell_id).unwrap();
        let cell = design.cell(cell_idx);

        assert_eq!(cell.params.len(), 3);

        let init = cell.params.get(&pool.intern("INIT")).unwrap();
        assert_eq!(init, &Property::bit_vector("0"));

        let inv = cell.params.get(&pool.intern("IS_C_INVERTED")).unwrap();
        assert_eq!(inv, &Property::bit_vector("1"));

        // "SLICE_X0Y0" contains non-binary characters, so it's a string
        let slice = cell.params.get(&pool.intern("SLICE")).unwrap();
        assert_eq!(slice, &Property::string("SLICE_X0Y0"));
    }

    // =========================================================================
    // Net naming test
    // =========================================================================

    #[test]
    fn test_netnames_rename_nets() {
        let pool = make_pool();
        let design = parse_json(BLINKY_JSON, &pool).unwrap();

        // After parsing, nets originally named $signal$2 and $signal$3
        // should be renamed to "clk" and "led" respectively
        let clk_id = pool.intern("clk");
        let led_id = pool.intern("led");

        let clk_net_idx = design.net_by_name(clk_id);
        assert!(clk_net_idx.is_some(), "Net 'clk' should exist after renaming");

        let led_net_idx = design.net_by_name(led_id);
        assert!(led_net_idx.is_some(), "Net 'led' should exist after renaming");

        // The actual NetInfo objects should have the renamed names
        let clk_net = design.net(clk_net_idx.unwrap());
        assert_eq!(clk_net.name, clk_id);

        let led_net = design.net(led_net_idx.unwrap());
        assert_eq!(led_net.name, led_id);
    }

    // =========================================================================
    // Hidden net name test
    // =========================================================================

    #[test]
    fn test_hidden_netnames_not_applied() {
        let json = r#"{
            "modules": {
                "top": {
                    "attributes": { "top": "00000000000000000000000000000001" },
                    "parameter_default_values": {},
                    "ports": {
                        "a": { "direction": "input", "bits": [2] }
                    },
                    "cells": {},
                    "netnames": {
                        "$internal_wire": {
                            "hide_name": 1,
                            "bits": [2],
                            "attributes": {}
                        }
                    }
                }
            }
        }"#;
        let pool = make_pool();
        let design = parse_json(json, &pool).unwrap();

        // The hidden name should not be applied
        let hidden_id = pool.intern("$internal_wire");
        assert!(
            design.net_by_name(hidden_id).is_none(),
            "Hidden net name should not be in lookup"
        );
    }

    // =========================================================================
    // Complex design test (two cells, shared nets)
    // =========================================================================

    #[test]
    fn test_two_cells_shared_net() {
        let json = r#"{
            "modules": {
                "top": {
                    "attributes": { "top": "00000000000000000000000000000001" },
                    "parameter_default_values": {},
                    "ports": {},
                    "cells": {
                        "src_cell": {
                            "hide_name": 0,
                            "type": "SRC",
                            "parameters": {},
                            "attributes": {},
                            "port_directions": { "O": "output" },
                            "connections": { "O": [42] }
                        },
                        "sink_cell": {
                            "hide_name": 0,
                            "type": "SINK",
                            "parameters": {},
                            "attributes": {},
                            "port_directions": { "I": "input" },
                            "connections": { "I": [42] }
                        }
                    },
                    "netnames": {
                        "wire_42": {
                            "hide_name": 0,
                            "bits": [42],
                            "attributes": {}
                        }
                    }
                }
            }
        }"#;
        let pool = make_pool();
        let design = parse_json(json, &pool).unwrap();

        // Both cells share net 42 (renamed to "wire_42")
        let wire_name = pool.intern("wire_42");
        let net_idx = design.net_by_name(wire_name).unwrap();
        let net = design.net(net_idx);

        // src_cell drives it
        let src_id = pool.intern("src_cell");
        let src_idx = design.cell_by_name(src_id).unwrap();
        assert_eq!(net.driver.cell, src_idx);

        // sink_cell uses it
        let sink_id = pool.intern("sink_cell");
        let sink_idx = design.cell_by_name(sink_id).unwrap();
        assert!(net.users.iter().any(|u| u.cell == sink_idx));
    }

    // =========================================================================
    // Netnames attributes test
    // =========================================================================

    #[test]
    fn test_netnames_attributes() {
        let json = r#"{
            "modules": {
                "top": {
                    "attributes": { "top": "00000000000000000000000000000001" },
                    "parameter_default_values": {},
                    "ports": {
                        "clk": { "direction": "input", "bits": [2] }
                    },
                    "cells": {},
                    "netnames": {
                        "clk": {
                            "hide_name": 0,
                            "bits": [2],
                            "attributes": {
                                "src": "design.v:1"
                            }
                        }
                    }
                }
            }
        }"#;
        let pool = make_pool();
        let design = parse_json(json, &pool).unwrap();

        let clk_id = pool.intern("clk");
        let net_idx = design.net_by_name(clk_id).unwrap();
        let net = design.net(net_idx);

        let src_key = pool.intern("src");
        let attr = net.attrs.get(&src_key).unwrap();
        assert_eq!(attr.as_str(), "design.v:1");
    }

    // =========================================================================
    // Property edge cases
    // =========================================================================

    #[test]
    fn test_property_empty_binary_string() {
        // Empty string that looks binary (all 0s and 1s) but is empty
        let val = serde_json::json!("");
        let prop = parse_property(&val).unwrap();
        // Empty string: all chars vacuously satisfy being 0/1, but is_empty check
        // makes it a string
        assert!(prop.is_string());
    }

    #[test]
    fn test_parse_property_unsupported_type() {
        let val = serde_json::json!(true);
        assert!(parse_property(&val).is_err());
    }
}
