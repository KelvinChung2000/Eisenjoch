//! The place-and-route state, captured for a viewer in the same process.
//!
//! A viewer and the flow's [`Context`] share one [`Live`]. The viewer calls
//! [`Live::request`] each time it wants a fresh picture and reads
//! [`Live::latest`]. The flow captures its state only after a request, at the
//! next iteration that a placer or router offers through
//! [`Context::offer_live`], so the viewer's request rate sets the capture rate
//! and an unwatched flow pays one atomic load per iteration. [`Live::cancel`]
//! stops the flow at its next offer.
//!
//! A frame names each placed cell's BEL as `tile/bel` and each routed net's
//! wires in the `ROUTING` attribute's `wire;pip;strength` form, so it applies
//! to the packed netlist that [`Live::latest`] returns beside it. A cell or
//! net that a frame leaves out is unplaced or unrouted.

use std::collections::BTreeMap;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Arc, Mutex, MutexGuard, PoisonError};

use crate::chipdb::{PipId, WireId};
use crate::common::PlaceStrength;
use crate::context::Context;
use crate::netlist::NetId;
use crate::routed_json::{bel_name, routed_json, routing, RoutedJsonError};

/// A net in a router's working plans: its wires, each with the pip that drives it.
pub type PlannedNet = (NetId, Vec<(WireId, Option<PipId>)>);

/// Where a frame reads routing from.
pub enum LiveRouting<'a> {
    /// The wires bound in the context.
    Bound,
    /// A router's working plans.
    Planned(&'a [PlannedNet]),
}

/// What the flow is doing when it offers its state.
pub struct LiveProgress<'a> {
    pub phase: &'a str,
    pub iteration: Option<usize>,
    pub overused: Option<usize>,
    pub routing: LiveRouting<'a>,
}

impl<'a> LiveProgress<'a> {
    /// The bound state of `phase`, at `iteration` when it has one.
    pub fn bound(phase: &'a str, iteration: Option<usize>) -> Self {
        Self {
            phase,
            iteration,
            overused: None,
            routing: LiveRouting::Bound,
        }
    }
}

/// One captured picture of the placement and routing.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct LiveFrame {
    /// Counts captures from 1, so a viewer can tell a new frame from one it drew.
    pub version: u64,
    pub phase: String,
    pub iteration: Option<usize>,
    /// Routing nodes that more than one net uses, while a router negotiates.
    pub overused: Option<usize>,
    pub cells: BTreeMap<String, String>,
    pub nets: BTreeMap<String, String>,
}

/// The latest frame and the packed netlist it places and routes.
#[derive(Debug, Clone)]
pub struct LiveSnapshot {
    pub netlist: Arc<serde_json::Value>,
    pub frame: Arc<LiveFrame>,
}

/// The flow stopped because its viewer cancelled it.
#[derive(Debug, Clone, Copy, PartialEq, Eq, thiserror::Error)]
#[error("place and route was cancelled")]
pub struct Cancelled;

/// The state a viewer and a running flow share.
#[derive(Default)]
pub struct Live {
    wanted: AtomicBool,
    cancelled: AtomicBool,
    captured: Mutex<Captured>,
}

#[derive(Default)]
struct Captured {
    netlist: Option<Arc<serde_json::Value>>,
    frame: Option<Arc<LiveFrame>>,
}

impl Live {
    pub fn new() -> Arc<Self> {
        Arc::default()
    }

    /// Ask the flow to capture its state at its next offer.
    pub fn request(&self) {
        self.wanted.store(true, Ordering::Release);
    }

    /// The newest capture, if the flow has made one.
    pub fn latest(&self) -> Option<LiveSnapshot> {
        let captured = self.lock();
        Some(LiveSnapshot {
            netlist: captured.netlist.clone()?,
            frame: captured.frame.clone()?,
        })
    }

    /// Stop the flow at its next offer or step.
    pub fn cancel(&self) {
        self.cancelled.store(true, Ordering::Release);
    }

    pub fn is_cancelled(&self) -> bool {
        self.cancelled.load(Ordering::Acquire)
    }

    fn wanted(&self) -> bool {
        self.wanted.load(Ordering::Acquire)
    }

    fn lock(&self) -> MutexGuard<'_, Captured> {
        self.captured.lock().unwrap_or_else(PoisonError::into_inner)
    }

    fn capture(&self, ctx: &Context, progress: &LiveProgress<'_>) -> Result<(), RoutedJsonError> {
        self.wanted.store(false, Ordering::Release);
        let netlist = match self.lock().netlist {
            Some(_) => None,
            None => Some(Arc::new(routed_json(ctx)?)),
        };
        let mut frame = frame(ctx, progress);
        let mut captured = self.lock();
        frame.version = captured.frame.as_ref().map_or(0, |last| last.version) + 1;
        captured.frame = Some(Arc::new(frame));
        if netlist.is_some() {
            captured.netlist = netlist;
        }
        Ok(())
    }
}

impl Context {
    /// Share this flow's state, and its cancellation, with a viewer.
    pub fn attach_live(&mut self, live: Arc<Live>) {
        self.live = Some(live);
    }

    /// Whether a viewer waits for a capture, so a router can skip building its plans otherwise.
    pub fn live_wanted(&self) -> bool {
        self.live.as_ref().is_some_and(|live| live.wanted())
    }

    /// Fails once the viewer has cancelled the flow.
    pub fn live_cancelled(&self) -> Result<(), Cancelled> {
        match self.live.as_ref().is_some_and(|live| live.is_cancelled()) {
            true => Err(Cancelled),
            false => Ok(()),
        }
    }

    /// Capture the state if a viewer asked for it; fails once the viewer has cancelled.
    pub fn offer_live(&self, progress: LiveProgress<'_>) -> Result<(), Cancelled> {
        self.live_cancelled()?;
        if self.live_wanted() {
            self.publish_live(progress);
        }
        Ok(())
    }

    /// Capture the state whether or not a viewer asked, as at the end of a phase.
    pub fn publish_live(&self, progress: LiveProgress<'_>) {
        if let Some(live) = &self.live {
            if let Err(error) = live.capture(self, &progress) {
                log::warn!("Live state not captured: {error}");
            }
        }
    }
}

fn frame(ctx: &Context, progress: &LiveProgress<'_>) -> LiveFrame {
    let cells = ctx
        .design
        .iter_alive_cells()
        .filter_map(|(_, cell)| Some((ctx.name_of(cell.name).to_owned(), bel_name(ctx, cell.bel?))))
        .collect();
    let nets = match progress.routing {
        LiveRouting::Bound => ctx
            .design
            .iter_alive_nets()
            .filter(|(_, net)| !net.wires.is_empty())
            .map(|(_, net)| {
                let wires = net
                    .wires
                    .iter()
                    .map(|(&wire, binding)| (wire, binding.pip, binding.strength));
                (ctx.name_of(net.name).to_owned(), routing(ctx, wires))
            })
            .collect(),
        LiveRouting::Planned(plans) => plans
            .iter()
            .filter(|(_, wires)| !wires.is_empty())
            .map(|(net, wires)| {
                let wires = wires
                    .iter()
                    .map(|&(wire, pip)| (wire, pip, PlaceStrength::Strong));
                let name = ctx.name_of(ctx.design.net(*net).name).to_owned();
                (name, routing(ctx, wires))
            })
            .collect(),
    };
    LiveFrame {
        version: 0,
        phase: progress.phase.to_owned(),
        iteration: progress.iteration,
        overused: progress.overused,
        cells,
        nets,
    }
}
