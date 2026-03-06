# Design Edit API

## Goal

Replace direct mutable access to `CellInfo`/`NetInfo` (`cell_mut()`/`net_mut()`) with a proxy-based edit API. External code (packer, placer, router, Python bindings) interacts only through view proxies (read) and editor proxies (write). No raw struct access leaks outside the `netlist`/`context` modules.

## Decisions

- **Editor proxy pattern**: `CellEditor`/`NetEditor` returned by `ctx.cell_edit(idx)`/`ctx.net_edit(idx)` with setter methods. No assignment-syntax macros (Rust lacks Python-style property setters).
- **Read-only DesignView proxy**: `ctx.design()` returns `DesignView` for design-level queries (`num_cells`, iteration, name lookups).
- **No DesignEditor proxy**: design-level mutations (`add_cell`, `add_net`, `remove_cell`, etc.) are direct methods on Context.
- **Packer refactored to `&mut Context`**: `packer_parts()` removed. Packer uses the same Context API as placer/router.
- **`CellInfo`/`NetInfo` stay as-is internally**: fields remain `pub` within the struct, but the structs and `Design` are not accessible from outside. Enforcement is architectural (nobody outside context/netlist gets `&Design`).
- **`design_mut()` / `cell_mut()` / `net_mut()` removed from public API**: `design_mut` removed entirely, `cell_mut`/`net_mut` become private to Design (used only inside editor methods).
- **Future threading**: editor proxies can swap `&mut CellInfo` for `RwLockWriteGuard<CellInfo>` without changing any caller.

## API Surface

### Design-level reads (DesignView proxy)

```rust
pub struct DesignView<'a> {
    design: &'a Design,
    pool: &'a IdStringPool,
}

ctx.design().num_cells() -> usize
ctx.design().num_nets() -> usize
ctx.design().iter_alive_cells() -> impl Iterator<Item = CellView>
ctx.design().iter_alive_nets() -> impl Iterator<Item = NetView>
ctx.design().cell_by_name(name) -> Option<CellIdx>
ctx.design().net_by_name(name) -> Option<NetIdx>
ctx.design().is_empty() -> bool
```

### Design-level mutations (direct on Context)

```rust
ctx.add_cell(name, cell_type) -> CellIdx
ctx.add_net(name) -> NetIdx
ctx.remove_cell(name)
ctx.remove_net(name)
ctx.rename_net(net_idx, new_name)
ctx.set_design(design)  // replaces entire design (frontend loading)
```

### Cell read (CellView -- already exists)

```rust
ctx.cell(idx) -> CellView

// CellView methods (existing + any needed additions):
cell_view.name() -> IdString
cell_view.cell_type() -> IdString
cell_view.bel() -> Option<BelView>
cell_view.bel_strength() -> PlaceStrength
cell_view.is_alive() -> bool
cell_view.ports() -> &FxHashMap<IdString, PortInfo>
cell_view.port(name) -> Option<&PortInfo>
cell_view.attrs() -> &FxHashMap<IdString, Property>
cell_view.params() -> &FxHashMap<IdString, Property>
cell_view.cluster() -> Option<CellIdx>
cell_view.region() -> Option<i32>
```

### Cell write (CellEditor)

```rust
pub struct CellEditor<'a> {
    cell: &'a mut CellInfo,
}

ctx.cell_edit(idx) -> CellEditor

// Placement
cell_editor.set_bel(bel: Option<BelId>, strength: PlaceStrength)

// Ports
cell_editor.add_port(name: IdString, port_type: PortType)
cell_editor.connect_port(port: IdString, net: NetIdx, user_idx: Option<u32>)
cell_editor.disconnect_port(port: IdString)
cell_editor.rename_port(old: IdString, new: IdString)

// Metadata
cell_editor.set_type(cell_type: IdString)
cell_editor.set_attr(key: IdString, value: Property)
cell_editor.set_param(key: IdString, value: Property)
cell_editor.set_cluster(root: Option<CellIdx>)
cell_editor.set_region(region: Option<i32>)
cell_editor.set_flat_index(idx: Option<FlatIndex>)
cell_editor.set_timing_index(idx: Option<TimingIndex>)
cell_editor.mark_dead()
```

### Net read (NetView -- already exists)

```rust
ctx.net(idx) -> NetView

net_view.name() -> IdString
net_view.driver() -> &PortRef
net_view.users() -> &[PortRef]
net_view.wires() -> &FxHashMap<WireId, PipMap>
net_view.is_alive() -> bool
net_view.has_driver() -> bool
net_view.num_users() -> usize
net_view.clock_constraint() -> DelayT
net_view.region() -> Option<i32>
```

### Net write (NetEditor)

```rust
pub struct NetEditor<'a> {
    net: &'a mut NetInfo,
}

ctx.net_edit(idx) -> NetEditor

// Connectivity
net_editor.set_driver(cell: Option<CellIdx>, port: IdString)
net_editor.clear_driver()
net_editor.add_user(cell: CellIdx, port: IdString) -> u32  // returns user_idx
net_editor.disconnect_user(user_idx: usize)

// Routing
net_editor.add_wire(wire: WireId, pip: Option<PipId>, strength: PlaceStrength)
net_editor.clear_wires()

// Metadata
net_editor.set_clock_constraint(period_ps: DelayT)
net_editor.set_region(region: Option<i32>)
net_editor.mark_dead()
```

### Hardware operations (unchanged)

```rust
ctx.bel(id) -> BelView
ctx.wire(id) -> WireView
ctx.pip(id) -> PipView
ctx.bind_bel(bel, cell_idx, strength) -> bool
ctx.unbind_bel(bel)
ctx.bind_wire(wire, net_idx, strength)    // pub(crate)
ctx.unbind_wire(wire)                      // pub(crate)
ctx.bind_pip(pip, net_idx, strength)       // pub(crate)
ctx.unbind_pip(pip)                        // pub(crate)
```

## Visibility Summary

```
Public:
  Context::design()      -> DesignView
  Context::cell(idx)     -> CellView
  Context::cell_edit(idx) -> CellEditor
  Context::net(idx)      -> NetView
  Context::net_edit(idx)  -> NetEditor
  Context::add_cell/add_net/remove_cell/remove_net/rename_net/set_design
  Context::bind_bel/unbind_bel

pub(crate):
  Context::bind_wire/unbind_wire/bind_pip/unbind_pip
  Design (struct itself)
  Design::cell(idx) -> &CellInfo
  Design::net(idx) -> &NetInfo
  Design::cell_edit(idx) -> CellEditor
  Design::net_edit(idx) -> NetEditor
  CellInfo/NetInfo (structs with pub fields, accessible within crate)

Private:
  Design::cell_mut(idx)   -- used only inside Design/editor methods
  Design::net_mut(idx)    -- used only inside Design/editor methods
  Context::design_mut()   -- removed
  Context::packer_parts() -- removed
```

## Enforcement

The packer, placer, and router receive `&mut Context`, never `&Design`. They cannot access `CellInfo` or `NetInfo` fields directly. They read through `CellView`/`NetView` and write through `CellEditor`/`NetEditor`.

The borrow pattern for typical usage:

```rust
// Read, then write (borrows don't overlap)
let cell_type = ctx.cell(idx).cell_type();  // &ctx dropped at semicolon
ctx.cell_edit(idx).set_type(new_type);       // &mut ctx

// Iterate, then mutate
let indices: Vec<CellIdx> = ctx.design().iter_alive_cells()
    .map(|cv| cv.idx())
    .collect();                              // DesignView dropped
for idx in indices {
    ctx.cell_edit(idx).set_type(new_type);
}

// Two-cell operations get dedicated methods
ctx.bind_bel(bel, cell_idx, strength);  // handles cell.bel + bel_to_cell atomically
```

## Packer Refactor

The packer changes from:

```rust
pub fn pack(design: &mut Design, chipdb: &ChipDb, pool: &IdStringPool, plugin: Option<...>)
```

To:

```rust
pub fn pack(ctx: &mut Context, plugin: Option<...>)
```

The packer helper functions (`connect_port`, `disconnect_port`, `rename_port`, etc.) become thin wrappers around `CellEditor`/`NetEditor` methods, or are removed entirely since the editor methods replace them.

## Deferred

- Thread-safe interior mutability (`RwLock<CellInfo>` backing for editors)
- Transactional edits (batch changes with rollback for speculative SA swaps)
