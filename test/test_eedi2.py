#!/usr/bin/env python3
"""Functional tests for the EEDI2 VapourSynth plugin.

Usage: python3 test/test_eedi2.py [path/to/libeedi2.so]

Without arguments the plugin is expected to be autoloaded (e.g. from the
installed wheel). Requires the vapoursynth Python module and numpy.
"""
import sys

import numpy as np
import vapoursynth as vs

core = vs.core
if len(sys.argv) > 1:
    core.std.LoadPlugin(sys.argv[1])

WIDTH, HEIGHT, FRAMES = 64, 48, 4


def noise_clip(fmt):
    rng = np.random.default_rng(1234)
    base = core.std.BlankClip(width=WIDTH, height=HEIGHT, length=FRAMES, format=fmt)
    peak = (1 << core.get_video_format(fmt).bits_per_sample) - 1

    def fill(n, f):
        f = f.copy()
        for p in range(f.format.num_planes):
            a = np.asarray(f[p])
            a[:] = rng.integers(0, peak + 1, size=a.shape)
        return f

    return core.std.ModifyFrame(base, base, fill)


def check(cond, msg):
    if not cond:
        print('FAIL:', msg)
        sys.exit(1)


def expect_error(fn, msg):
    try:
        fn()
    except vs.Error:
        return
    check(False, msg)


for fmt in (vs.YUV420P8, vs.YUV422P10, vs.YUV444P16, vs.GRAY8, vs.GRAY12):
    src = noise_clip(fmt)
    for field in range(4):
        for pp in range(4):
            out = core.eedi2.EEDI2(src, field=field, pp=pp)
            check(out.width == WIDTH and out.height == HEIGHT * 2, f'size {fmt.name} field={field} pp={pp}')
            check(out.format == src.format and out.num_frames == FRAMES, f'format {fmt.name}')
            for n in range(FRAMES):
                out.get_frame(n)

    # Original lines are copied unchanged to the requested field.
    for field in (0, 1):
        out = core.eedi2.EEDI2(src, field=field)
        a = np.asarray(out.get_frame(0)[0])
        b = np.asarray(src.get_frame(0)[0])
        check(np.array_equal(a[1 - field::2], b), f'field copy {fmt.name} field={field}')

    for m, h in ((1, HEIGHT), (2, HEIGHT), (3, HEIGHT * 2)):
        out = core.eedi2.EEDI2(src, field=1, map=m)
        check(out.height == h, f'map={m} height {fmt.name}')
        out.get_frame(0)

expect_error(lambda: core.eedi2.EEDI2(core.std.BlankClip(format=vs.RGB24), field=1).get_frame(0), 'RGB must fail')
expect_error(lambda: core.eedi2.EEDI2(core.std.BlankClip(format=vs.YUV420P8), field=4).get_frame(0), 'field=4 must fail')
expect_error(lambda: core.eedi2.EEDI2(core.std.BlankClip(format=vs.YUV420P8), field=1, maxd=30).get_frame(0), 'maxd=30 must fail')

print('All EEDI2 tests passed')
