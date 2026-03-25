use super::sa::PlacerSaCfg;
use super::heap::PlacerHeapCfg;
use super::opt_trans_place::OptTransPlacerCfg;
use super::electro_place::ElectroPlaceCfg;

/// Unified placer selection with embedded configuration.
#[derive(Debug, Clone)]
pub enum PlacerChoice {
    Sa(PlacerSaCfg),
    Heap(PlacerHeapCfg),
    OptTrans(OptTransPlacerCfg),
    Electro(ElectroPlaceCfg),
}

impl PlacerChoice {
    /// Get the seed from whichever config variant is selected.
    pub fn seed(&self) -> u64 {
        match self {
            Self::Sa(c) => c.seed,
            Self::Heap(c) => c.seed,
            Self::OptTrans(c) => c.seed,
            Self::Electro(c) => c.seed,
        }
    }
}

impl Default for PlacerChoice {
    fn default() -> Self {
        Self::Heap(PlacerHeapCfg::default())
    }
}
