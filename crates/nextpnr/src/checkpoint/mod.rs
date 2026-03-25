//! Checkpoint save/load/restore for incremental place-and-route.
//!
//! Users manage the incremental flow explicitly:
//!
//! ```ignore
//! // Save after a successful P&R run:
//! checkpoint::save(&ctx, &path)?;
//!
//! // In a later session, restore and re-place/re-route:
//! let cp = Checkpoint::load(&path)?;
//! let report = checkpoint::restore(&mut ctx, &cp)?;
//! // Restored cells are Fixed, so normal place()/route() skips them.
//! placer.place(&mut ctx, &cfg)?;
//! router.route(&mut ctx, &cfg)?;
//! ```

mod diff;
mod save;
mod types;
pub mod restore;

pub use types::*;
pub use save::{save, build_checkpoint};
pub use diff::diff_by_name;
pub use restore::{compute_fingerprint, restore, RestoreReport};
