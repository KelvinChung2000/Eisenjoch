use crate::common::IdString;
use crate::context::Context;

pub struct IdStringView<'a> {
    ctx: &'a Context,
    id: IdString,
}

impl<'a> IdStringView<'a> {
    pub(crate) fn new(ctx: &'a Context, id: IdString) -> Self {
        Self { ctx, id }
    }

    #[inline]
    pub fn id(&self) -> IdString {
        self.id
    }

    #[inline]
    pub fn index(&self) -> i32 {
        self.id.index()
    }

    #[inline]
    pub fn is_empty(&self) -> bool {
        self.id.is_empty()
    }

    #[inline]
    pub fn as_str(&self) -> &'a str {
        self.ctx.name_of(self.id)
    }
}

macro_rules! define_view {
    ($name:ident, $id_type:ty) => {
        #[derive(Clone, Copy)]
        pub struct $name<'a> {
            pub(super) ctx: &'a Context,
            pub(super) id: $id_type,
        }

        impl<'a> $name<'a> {
            pub(crate) fn new(ctx: &'a Context, id: $id_type) -> Self {
                Self { ctx, id }
            }

            #[inline]
            pub fn id(&self) -> $id_type { self.id }
        }

        impl std::ops::Deref for $name<'_> {
            type Target = $id_type;
            #[inline]
            fn deref(&self) -> &$id_type { &self.id }
        }

        impl PartialEq for $name<'_> {
            fn eq(&self, other: &Self) -> bool { self.id == other.id }
        }
        impl Eq for $name<'_> {}

        impl std::hash::Hash for $name<'_> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.id.hash(state); }
        }

        impl PartialEq<$id_type> for $name<'_> {
            fn eq(&self, other: &$id_type) -> bool { self.id == *other }
        }
        impl PartialEq<$name<'_>> for $id_type {
            fn eq(&self, other: &$name<'_>) -> bool { *self == other.id }
        }

        impl From<$name<'_>> for $id_type {
            fn from(v: $name<'_>) -> $id_type { v.id }
        }
        impl From<&$name<'_>> for $id_type {
            fn from(v: &$name<'_>) -> $id_type { v.id }
        }
    };
}

macro_rules! define_hardware_view {
    ($name:ident, $id_type:ty) => {
        #[derive(Clone, Copy)]
        pub struct $name<'a> {
            pub(super) ctx: &'a Context,
            pub(super) id: $id_type,
        }

        impl<'a> $name<'a> {
            pub(crate) fn new(ctx: &'a Context, id: $id_type) -> Self {
                Self { ctx, id }
            }

            #[inline]
            pub fn id(&self) -> $id_type { self.id }
        }

        impl std::ops::Deref for $name<'_> {
            type Target = $id_type;
            #[inline]
            fn deref(&self) -> &$id_type { &self.id }
        }

        impl PartialEq for $name<'_> {
            fn eq(&self, other: &Self) -> bool { self.id == other.id }
        }
        impl Eq for $name<'_> {}

        impl std::hash::Hash for $name<'_> {
            fn hash<H: std::hash::Hasher>(&self, state: &mut H) { self.id.hash(state); }
        }

        impl PartialEq<$id_type> for $name<'_> {
            fn eq(&self, other: &$id_type) -> bool { self.id == *other }
        }
        impl PartialEq<$name<'_>> for $id_type {
            fn eq(&self, other: &$name<'_>) -> bool { *self == other.id }
        }

        impl From<$name<'_>> for $id_type {
            fn from(v: $name<'_>) -> $id_type { v.id }
        }
        impl From<&$name<'_>> for $id_type {
            fn from(v: &$name<'_>) -> $id_type { v.id }
        }

        impl std::fmt::Display for $name<'_> {
            fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                self.id.fmt(f)
            }
        }
    };
}

pub(super) use define_hardware_view;
pub(super) use define_view;
