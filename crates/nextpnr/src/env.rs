//! Strict reads of the `NPNR_*` tuning knobs.

use std::env::VarError;
use std::fmt::Display;
use std::str::FromStr;

/// Parse `raw` as the type the call site asks for, naming the variable, the
/// value and the type on failure. An empty `raw` is a failure.
pub fn parse_var<T>(name: &str, raw: &str) -> T
where
    T: FromStr,
    T::Err: Display,
{
    assert!(
        !raw.is_empty(),
        "{name} is set but empty. Give it a {} value, or unset it to take the default.",
        std::any::type_name::<T>()
    );
    raw.parse().unwrap_or_else(|e| {
        panic!(
            "{name}={raw:?} is not a {}: {e}",
            std::any::type_name::<T>()
        )
    })
}

/// Read `name`, or `None` when it is unset. A value that does not parse is a
/// configuration error and panics, so a mistyped knob cannot pass as a run
/// with the knob at its default.
pub fn var<T>(name: &str) -> Option<T>
where
    T: FromStr,
    T::Err: Display,
{
    match std::env::var(name) {
        Ok(raw) => Some(parse_var(name, &raw)),
        Err(VarError::NotPresent) => None,
        Err(VarError::NotUnicode(raw)) => panic!(
            "{name} holds bytes that are not UTF-8 ({raw:?}). Give it a {} value.",
            std::any::type_name::<T>()
        ),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn panic_message<T>(name: &str, raw: &str) -> String
    where
        T: FromStr,
        T::Err: Display,
    {
        let payload = std::panic::catch_unwind(|| parse_var::<T>(name, raw))
            .err()
            .expect("parse_var should panic on a malformed value");
        payload
            .downcast_ref::<String>()
            .expect("the panic payload should be a formatted String")
            .clone()
    }

    #[test]
    fn parse_var_accepts_the_values_the_knobs_take() {
        assert_eq!(parse_var::<usize>("NPNR_OT_MAX_ITERS", "20"), 20);
        assert_eq!(parse_var::<u64>("NPNR_OT_SEED", "0"), 0);
        assert_eq!(parse_var::<i32>("NPNR_OT_CORRIDOR_HALO_MAX", "-3"), -3);
        assert_eq!(parse_var::<f64>("NPNR_OT_BPR_CAP", "1.2"), 1.2);
        assert_eq!(parse_var::<f64>("NPNR_OT_BPR_CAP", "inf"), f64::INFINITY);
    }

    #[test]
    fn parse_var_panics_naming_the_variable_and_the_value() {
        let cases = [
            ("", "is set but empty"),
            ("12.5", "NPNR_OT_MAX_ITERS=\"12.5\" is not a usize"),
            ("twenty", "NPNR_OT_MAX_ITERS=\"twenty\" is not a usize"),
            (" 20", "NPNR_OT_MAX_ITERS=\" 20\" is not a usize"),
            ("20 ", "NPNR_OT_MAX_ITERS=\"20 \" is not a usize"),
            ("-1", "NPNR_OT_MAX_ITERS=\"-1\" is not a usize"),
        ];
        for (raw, expected) in cases {
            let message = panic_message::<usize>("NPNR_OT_MAX_ITERS", raw);
            assert!(
                message.contains(expected),
                "message {message:?} does not contain {expected:?}"
            );
        }
    }

    #[test]
    fn parse_var_names_a_float_knob_as_f64() {
        let message = panic_message::<f64>("NPNR_OT_STEINER", "1,");
        assert!(
            message.contains("NPNR_OT_STEINER=\"1,\" is not a f64"),
            "{message}"
        );
    }
}
