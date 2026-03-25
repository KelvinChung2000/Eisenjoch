use super::maze::Router1Cfg;
use super::negotiation::Router2Cfg;

/// Unified router selection with embedded configuration.
#[derive(Debug, Clone)]
pub enum RouterChoice {
    Maze(Router1Cfg),
    Negotiation(Router2Cfg),
}

impl Default for RouterChoice {
    fn default() -> Self {
        Self::Maze(Router1Cfg::default())
    }
}
