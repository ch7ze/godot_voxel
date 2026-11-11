//! Fixed-point conversion helpers ported from `storage/funcs.h:123-152`.

/// Converts an int8 value into a float in [-1, 1]. -128 clamps to -1.
/// Source: storage/funcs.h:126-128
pub fn s8_to_snorm(v: i8) -> f32 {
    (v as f32 / 127.0).max(-1.0)
}

/// Source: storage/funcs.h:129-131
pub fn s8_to_snorm_noclamp(v: i8) -> f32 {
    v as f32 / 127.0
}

/// Converts a float in [-1, 1] to an int8, clamping out-of-range input.
/// Source: storage/funcs.h:135-137
pub fn snorm_to_s8(v: f32) -> i8 {
    (v.clamp(-1.0, 1.0) * 127.0) as i8
}

/// Converts an int16 value into a float in [-1, 1]. -32768 clamps to -1.
/// Source: storage/funcs.h:141-143
pub fn s16_to_snorm(v: i16) -> f32 {
    (v as f32 / 32767.0).max(-1.0)
}

/// Source: storage/funcs.h:144-146
pub fn s16_to_snorm_noclamp(v: i16) -> f32 {
    v as f32 / 32767.0
}

/// Converts a float in [-1, 1] to an int16, clamping out-of-range input.
/// Source: storage/funcs.h:150-152
pub fn snorm_to_s16(v: f32) -> i16 {
    (v.clamp(-1.0, 1.0) * 32767.0) as i16
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn s8_round_trip_is_exact_for_zero() {
        assert_eq!(s8_to_snorm(snorm_to_s8(0.0)), 0.0);
    }

    #[test]
    fn s8_clamps_below_minus_one() {
        // -128 has no exact counterpart and is clamped, per storage/funcs.h:125
        assert_eq!(s8_to_snorm(-128), -1.0);
    }

    #[test]
    fn s16_clamps_below_minus_one() {
        assert_eq!(s16_to_snorm(-32768), -1.0);
    }

    #[test]
    fn snorm_to_s8_saturates() {
        assert_eq!(snorm_to_s8(5.0), 127);
        assert_eq!(snorm_to_s8(-5.0), -127);
    }
}
