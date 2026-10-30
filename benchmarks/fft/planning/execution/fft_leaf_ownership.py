"""Immutable tile/unit/SPAD ownership, independent of FFT arithmetic/codegen.

Indices are window-local. A global batch base must never enter this resolver.
Different stages may assign different workers, but must reuse the same owner.
"""
from dataclasses import dataclass


def _integer(name, value, minimum=0):
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ValueError(f'{name} must be an integer >= {minimum}')
    return value


@dataclass(frozen=True)
class LeafOwner:
    replica: int
    batch: int
    leaf: int
    tile: int
    leaf_lane: int
    unit: int
    tile_slot: int
    leaf_slot: int


@dataclass(frozen=True)
class RegularSIMDJob:
    job_id: int
    worker: int
    leaf_lane: int
    butterfly_ids: tuple[int, ...]


@dataclass(frozen=True)
class LeafOwnershipPlan:
    leaves_per_batch: int
    leaf_length: int
    group_leaves: int
    window_batch: int
    num_units: int = 28
    workers_per_unit: int = 8
    bank_names: tuple[str, ...] = ('buf_a', 'buf_b')
    group_order: str = 'batch-major'

    def __post_init__(self):
        for name in ('leaves_per_batch', 'leaf_length', 'group_leaves',
                     'window_batch', 'num_units', 'workers_per_unit'):
            _integer(name, getattr(self, name), 1)
        if self.group_leaves not in (1, 2, 4):
            raise ValueError('group_leaves must be 1, 2 or 4')
        if self.leaves_per_batch % self.group_leaves:
            raise ValueError('leaf groups must not cross batch boundaries')
        if self.group_order != 'batch-major':
            raise ValueError('only batch-major ownership is supported')
        if (not isinstance(self.bank_names, tuple) or not self.bank_names
                or any(not isinstance(x, str) or not x for x in self.bank_names)
                or len(set(self.bank_names)) != len(self.bank_names)):
            raise ValueError('bank_names must be a nonempty tuple of unique names')

    @property
    def tiles_per_batch(self):
        return self.leaves_per_batch // self.group_leaves

    @property
    def max_tiles(self):
        return self.window_batch * self.tiles_per_batch

    @property
    def slots_per_unit(self):
        return ((self.max_tiles + self.num_units - 1) // self.num_units) * self.group_leaves

    @property
    def bank_elements(self):
        return self.slots_per_unit * 2 * self.leaf_length

    @property
    def total_elements(self):
        return len(self.bank_names) * self.bank_elements

    @property
    def arena_bytes(self):
        return 4 * self.total_elements

    def live_tiles(self, live_batch=None):
        live = self.window_batch if live_batch is None else _integer('live_batch', live_batch, 1)
        if live > self.window_batch:
            raise ValueError('live_batch exceeds the planned window')
        return live * self.tiles_per_batch

    def resolve(self, replica, *, live_batch=None):
        _integer('replica', replica)
        if replica >= self.live_tiles(live_batch) * self.group_leaves:
            raise ValueError('replica is outside the live window')
        batch, leaf = divmod(replica, self.leaves_per_batch)
        tile = batch * self.tiles_per_batch + leaf // self.group_leaves
        tile_slot, unit = divmod(tile, self.num_units)
        lane = leaf % self.group_leaves
        return LeafOwner(replica, batch, leaf, tile, lane, unit,
                         tile_slot, self.group_leaves * tile_slot + lane)

    def replicas_for_tile(self, tile, *, live_batch=None):
        _integer('tile', tile)
        if tile >= self.live_tiles(live_batch):
            raise ValueError('tile is outside the live window')
        batch, group = divmod(tile, self.tiles_per_batch)
        first = batch * self.leaves_per_batch + group * self.group_leaves
        return tuple(range(first, first + self.group_leaves))

    def tiles_for_unit(self, unit, *, live_batch=None):
        _integer('unit', unit)
        if unit >= self.num_units:
            raise ValueError('unit is outside the target')
        return range(unit, self.live_tiles(live_batch), self.num_units)

    def physical_unit(self, replica, base_unit, *, live_batch=None):
        _integer('base_unit', base_unit)
        if base_unit >= self.num_units:
            raise ValueError('base_unit is outside the target')
        return (base_unit + self.resolve(replica, live_batch=live_batch).unit) % self.num_units

    def resident_index(self, replica, element, *, bank, plane, live_batch=None):
        _integer('element', element)
        _integer('plane', plane)
        if element >= self.leaf_length or plane not in (0, 1):
            raise ValueError('resident element/plane is outside the leaf')
        if bank not in self.bank_names:
            raise ValueError(f'unknown resident bank {bank!r}')
        owner = self.resolve(replica, live_batch=live_batch)
        return (self.bank_names.index(bank) * self.bank_elements
                + owner.leaf_slot * 2 * self.leaf_length
                + plane * self.leaf_length + element)

    def regular_jobs(self, butterfly_count, compute_lanes):
        _integer('butterfly_count', butterfly_count, 1)
        _integer('compute_lanes', compute_lanes, 1)
        if compute_lanes not in (1, 2, 4) or butterfly_count % compute_lanes:
            raise ValueError('ordinary stage width must divide its butterfly count and be 1, 2 or 4')
        per_leaf = butterfly_count // compute_lanes
        result = []
        for leaf_lane in range(self.group_leaves):
            for original_job in range(per_leaf):
                job = leaf_lane * per_leaf + original_job
                start = original_job * compute_lanes
                result.append(RegularSIMDJob(job, job % self.workers_per_unit,
                    leaf_lane, tuple(range(start, start + compute_lanes))))
        return tuple(result)
