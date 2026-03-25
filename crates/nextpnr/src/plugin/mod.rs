//! Plugin system for the nextpnr-rust FPGA place-and-route tool.
//!
//! This module provides trait definitions for the three main CAD stages:
//! packing, placement, and routing. Plugins can implement these traits to
//! customize behavior. A [`PluginManager`] holds the active plugin for each
//! stage (defaulting to no-op implementations) and provides stub methods
//! for future native shared-library and Python plugin loading.

pub mod manager;
pub mod traits;

pub use manager::{PluginContext, PluginError, PluginManager};
pub use traits::{
    DefaultPacker, DefaultPlacerHooks, DefaultRouterHooks, PackerPlugin, PlacerPlugin,
    RouterPlugin,
};
