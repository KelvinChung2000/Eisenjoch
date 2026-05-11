#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct FlatIndex(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct TimingIndex(pub u32);

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct CellId(u32);

impl CellId {
    pub const NONE: Self = Self(u32::MAX);

    #[inline]
    pub(crate) const fn new(slot: u32, generation: u8) -> Self {
        Self(((generation as u32) << 24) | (slot & 0x00FF_FFFF))
    }

    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }

    #[inline]
    pub(crate) const fn slot(self) -> u32 {
        self.0 & 0x00FF_FFFF
    }

    #[inline]
    pub(crate) const fn generation(self) -> u8 {
        (self.0 >> 24) as u8
    }

    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    #[inline]
    pub const fn is_some(self) -> bool {
        self.0 != u32::MAX
    }
}

#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug, PartialOrd, Ord)]
pub struct NetId(u32);

impl NetId {
    pub const NONE: Self = Self(u32::MAX);

    #[inline]
    pub(crate) const fn new(slot: u32, generation: u8) -> Self {
        Self(((generation as u32) << 24) | (slot & 0x00FF_FFFF))
    }

    #[inline]
    pub const fn from_raw(raw: u32) -> Self {
        Self(raw)
    }

    #[inline]
    pub const fn raw(self) -> u32 {
        self.0
    }

    #[inline]
    pub(crate) const fn slot(self) -> u32 {
        self.0 & 0x00FF_FFFF
    }

    #[inline]
    pub(crate) const fn generation(self) -> u8 {
        (self.0 >> 24) as u8
    }

    #[inline]
    pub const fn is_none(self) -> bool {
        self.0 == u32::MAX
    }

    #[inline]
    pub const fn is_some(self) -> bool {
        self.0 != u32::MAX
    }
}
