/**
 * Numerical formatting utilities for Neural Trail
 * Handles decimal fractions (0.1667), normal percentages (16.67),
 * and anomalous scaled percentages (>100 e.g. 1667) safely.
 */
export function formatAccuracy(val: number | undefined | null): string {
  if (val == null || isNaN(val)) return '0.0%';
  let num = val;
  // If decimal fraction <= 1.0 (e.g. 0.1667 -> 16.67%)
  if (Math.abs(num) <= 1.0 && num !== 0) {
    num = num * 100;
  } else if (Math.abs(num) > 100) {
    // If value was scaled or multiplied twice (e.g. 1667 or 1667.0 -> 16.67%)
    num = num / 100;
  }
  return `${num.toFixed(1)}%`;
}

export function formatFlipRate(val: number | undefined | null): string {
  return formatAccuracy(val);
}
