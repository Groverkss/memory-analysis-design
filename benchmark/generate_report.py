#!/usr/bin/env python3
"""Generate a human-readable benchmark report from raw JSON data."""

import argparse
import json
from pathlib import Path


def categorize(name):
    """Assign a kernel to a category based on its filename."""
    categories = [
        ('rms_norm', 'Normalization'),
        ('layernorm', 'Normalization'),
        ('groupnorm', 'Normalization'),
        ('fused_residual', 'Normalization'),
        ('softmax', 'Softmax'),
        ('silu', 'Elementwise'),
        ('swiglu', 'Elementwise'),
        ('geglu', 'Elementwise'),
        ('elementwise', 'Elementwise'),
        ('rope', 'Elementwise'),
        ('transpose', 'Transpose'),
        ('outer_reduction', 'Outer Reduction'),
        ('outer_max', 'Outer Reduction'),
        ('inner_reduction', 'Inner Reduction'),
        ('reduction_3d', 'Inner Reduction'),
        ('global_reduction', 'Inner Reduction'),
        ('max_reduction', 'Inner Reduction'),
        ('matvec', 'MatVec'),
        ('cross_entropy', 'Fused'),
        ('argmax', 'Other'),
        ('row_sum', 'Other'),
    ]
    for prefix, cat in categories:
        if prefix in name:
            return cat
    return 'Other'


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--raw', required=True)
    parser.add_argument('--target', required=True)
    parser.add_argument('--peak-bw', type=float, default=0)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()

    data = json.loads(Path(args.raw).read_text())
    if not data:
        Path(args.output).write_text('No benchmark results.\n')
        return

    # Sort by category then name.
    for entry in data:
        entry['category'] = categorize(entry['name'])
    data.sort(key=lambda e: (e['category'], e['name']))

    lines = []
    lines.append(f'Benchmark Report — {args.target}')
    lines.append('=' * 80)
    if args.peak_bw > 0:
        lines.append(f'Peak memory bandwidth: {args.peak_bw:.1f} GB/s')
    lines.append('')

    # Table header.
    if args.peak_bw > 0:
        header = f'{"Kernel":<50} {"Time (us)":>10} {"BW (GB/s)":>10} {"% Peak":>8}'
    else:
        header = f'{"Kernel":<50} {"Time (us)":>10} {"BW (GB/s)":>10}'
    sep = '-' * len(header)

    current_cat = None
    for entry in data:
        if entry['category'] != current_cat:
            current_cat = entry['category']
            lines.append('')
            lines.append(f'  {current_cat}')
            lines.append(f'  {sep}')
            lines.append(f'  {header}')
            lines.append(f'  {sep}')

        time_us = entry['time_ns'] / 1000
        bw = entry['bw_gbps']
        pct = entry.get('pct_peak')

        if args.peak_bw > 0 and pct is not None:
            lines.append(f'  {entry["name"]:<50} {time_us:>10.1f} {bw:>10.1f} {pct:>7.0f}%')
        else:
            lines.append(f'  {entry["name"]:<50} {time_us:>10.1f} {bw:>10.1f}')

    # Summary.
    lines.append('')
    lines.append(sep)
    bws = [e['bw_gbps'] for e in data]
    lines.append(f'Total kernels benchmarked: {len(data)}')
    lines.append(f'Bandwidth range: {min(bws):.1f} — {max(bws):.1f} GB/s')
    if args.peak_bw > 0:
        pcts = [e['pct_peak'] for e in data if e.get('pct_peak') is not None]
        if pcts:
            lines.append(f'Peak utilization range: {min(pcts):.0f}% — {max(pcts):.0f}%')
            median_pct = sorted(pcts)[len(pcts) // 2]
            lines.append(f'Median peak utilization: {median_pct:.0f}%')
    lines.append('')

    report = '\n'.join(lines)
    Path(args.output).write_text(report)
    print(report)


if __name__ == '__main__':
    main()
