//! Shared metric computations for the nextpnr FPGA place-and-route tool.

pub mod bbox;
pub mod congestion;
pub mod density;
pub mod utilization;
pub mod wirelength;

pub use bbox::{compute_bbox, BoundingBox};
pub use congestion::{
    accumulate_edge_crossings, bresenham_line, compute_congestion_ratios, estimate_congestion,
    Axis, CongestionRatios, CongestionReport,
};
pub use density::{compute_sliding_window_density, placement_density, DensityReport};
pub use utilization::{utilization_report, ResourceRow, UtilizationReport};
pub use wirelength::{
    net_hpwl, net_line_estimate, total_hpwl, total_line_estimate, total_routed_wirelength,
};
