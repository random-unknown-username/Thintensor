//! Byte unit parsing and formatting.

use anyhow::{Result, bail};

pub fn parse_bytes(input: &str) -> Result<u64> {
    let trimmed = input.trim();
    if trimmed.is_empty() {
        bail!("byte value is empty");
    }

    let split = trimmed
        .find(|ch: char| !ch.is_ascii_digit() && ch != '.')
        .unwrap_or(trimmed.len());
    let (number, unit) = trimmed.split_at(split);
    if number.is_empty() {
        bail!("byte value is missing a number");
    }

    let value: f64 = number.parse()?;
    if !value.is_finite() || value < 0.0 {
        bail!("byte value must be finite and non-negative");
    }

    let multiplier = match unit.trim().to_ascii_lowercase().as_str() {
        "" | "b" => 1.0,
        "k" | "kb" | "kib" => 1024.0,
        "m" | "mb" | "mib" => 1024.0 * 1024.0,
        "g" | "gb" | "gib" => 1024.0 * 1024.0 * 1024.0,
        "t" | "tb" | "tib" => 1024.0 * 1024.0 * 1024.0 * 1024.0,
        other => bail!("unsupported byte unit {other}"),
    };

    let bytes = value * multiplier;
    if bytes > u64::MAX as f64 {
        bail!("byte value overflows u64");
    }

    Ok(bytes.round() as u64)
}

pub fn format_bytes(bytes: u64) -> String {
    const GIB: f64 = 1024.0 * 1024.0 * 1024.0;
    const MIB: f64 = 1024.0 * 1024.0;
    const KIB: f64 = 1024.0;

    let value = bytes as f64;
    if value >= GIB {
        format!("{:.2} GiB", value / GIB)
    } else if value >= MIB {
        format!("{:.2} MiB", value / MIB)
    } else if value >= KIB {
        format!("{:.2} KiB", value / KIB)
    } else {
        format!("{bytes} B")
    }
}
