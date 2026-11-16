"""Input-frozen DEVELOPMENT fixtures for bounded worm escape-horizon study.

These synthetic 80x80 permissions and soft guides are not native Worlds. The
caller must not use guidance to grant/revoke hard permission. No experiment
outcome or score is stored here; keep this input module stable after freeze.
"""
from __future__ import annotations

from worm_sketch import canonical, digest, environment
from worm_scent_study import guide_for

VERSION = 'd2.worm-horizon-inputs/1'
SCENES = ('open', 'slats', 'ring', 'canals', 'baffled', 'crucible')
SEEDS = (23, 79)
SIZE = (80, 80)
ORIGINS = {'open': (28, 38), 'slats': (23, 42), 'ring': (22, 36),
           'canals': (23, 42), 'baffled': (28, 38), 'crucible': (23, 42)}


def fixture(scene: str, seed: int):
    if scene not in SCENES or type(seed) is not int or seed not in SEEDS:
        raise ValueError('unregistered preregistered scene/seed')
    w, h = SIZE
    if scene in ('open', 'slats', 'ring'):
        legal = environment(w, h, scene)
    else:
        legal = environment(w, h, 'open')
        if scene == 'canals':
            for x in (32, 44, 56):
                for y in range(2, 78):
                    gap = (range(20, 32) if x in (32, 56) else range(48, 60))
                    if y not in gap:
                        legal.discard((x, y))
        elif scene == 'baffled':
            for y in (27, 39, 51, 63):
                for x in range(11, 69):
                    gap = range(16, 26) if y in (27, 51) else range(51, 61)
                    if x not in gap:
                        legal.discard((x, y))
        else:  # crucible: two thick, obstacle islands and a loose U rim
            legal -= {(x, y) for x in range(34, 49) for y in range(27, 56)}
            legal -= {(x, y) for x in range(13, 17) for y in range(19, 60)}
            legal -= {(x, y) for x in range(13, 65) for y in range(60, 64)
                      if not (23 <= x < 31)}
            legal -= {(x, y) for x in range(60, 65) for y in range(24, 63)}
    origin = ORIGINS[scene]
    assert origin in legal
    guide = guide_for(w, h, seed, 'attract')
    spec = dict(version=VERSION, scene=scene, seed=seed, canvas=[w, h],
                origin=list(origin), initial_heading=0, steps=192,
                pen_rhythm=dict(period=37, lifted=4),
                permitted_sha256=digest(canonical(sorted(map(list, legal)))),
                guide_u16_sha256=digest(canonical([guide[x, y] for y in range(h) for x in range(w)])))
    return spec, legal, guide


def preregistration():
    return dict(version=VERSION, qualification='DEVELOPMENT_INPUT_FROZEN',
                intended_arms=['strong_x16', 'short_escape', 'sustained_escape',
                               'sham_sustained', 'always_novel'],
                primary_metrics=['unique_cursor_cells', 'repeats', 'painted_cells'],
                secondary_metrics=['turns', 'escape_steps', 'escape_triggers', 'guide_advantage_u16'],
                cases=[fixture(s, k)[0] for s in SCENES for k in SEEDS])
