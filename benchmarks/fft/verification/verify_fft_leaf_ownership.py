"""Ownership contracts, independent queue allocation and resident lifetime."""
from dataclasses import FrozenInstanceError
import unittest
from planning.execution.fft_leaf_ownership import LeafOwnershipPlan


def reference_owners(plan, live_batch):
    """Allocate whole consecutive tiles to unit queues without resolver formulas."""
    queues = [[] for _ in range(plan.num_units)]
    result = {}
    turn = 0
    for batch in range(live_batch):
        for first in range(0,plan.leaves_per_batch,plan.group_leaves):
            unit = turn % plan.num_units
            for leaf in range(first,first+plan.group_leaves):
                replica = batch*plan.leaves_per_batch+leaf
                result[replica] = unit,len(queues[unit])
                queues[unit].append(replica)
            turn += 1
    return result, max(map(len,queues))


class LeafOwnershipTests(unittest.TestCase):
    def test_owner_and_resident_regions_against_unit_queues(self):
        for m,length,c,w in ((256,64,4,1),(256,64,4,4),(64,256,2,1),
                             (64,256,4,4),(128,64,4,2)):
            p = LeafOwnershipPlan(m,length,c,w)
            for live in {1,w}:
                expected,slots = reference_owners(p,live)
                all_addresses = set()
                for replica,(unit,slot) in expected.items():
                    owner = p.resolve(replica,live_batch=live)
                    self.assertEqual((owner.unit,owner.leaf_slot),(unit,slot))
                    for bank in p.bank_names:
                        for plane in (0,1):
                            for i in range(length):
                                off=p.resident_index(replica,i,bank=bank,plane=plane,live_batch=live)
                                key=unit,off
                                self.assertNotIn(key,all_addresses)
                                all_addresses.add(key)
                                self.assertLess(off,p.total_elements)
                    for base in (0,1,27):
                        self.assertEqual(p.physical_unit(replica,base,live_batch=live),(unit+base)%28)
                if live==w:
                    self.assertEqual(p.slots_per_unit,slots)

    def test_tiles_and_workers_cover_all_work(self):
        p=LeafOwnershipPlan(64,256,2,4)
        seen=[]
        for unit in range(28):
            for tile in p.tiles_for_unit(unit):
                for replica in p.replicas_for_tile(tile):
                    self.assertEqual(p.resolve(replica).unit,unit)
                    seen.append(replica)
        self.assertEqual(sorted(seen),list(range(256)))
        for count,width in ((8,4),(32,2),(64,4)):
            jobs=p.regular_jobs(count,width)
            actual=[]
            for worker in range(8):
                for job in jobs:
                    if job.worker==worker:
                        actual.extend((job.leaf_lane,i) for i in job.butterfly_ids)
            self.assertEqual(sorted(actual),[(leaf,i) for leaf in range(2) for i in range(count)])

    def test_partial_window_cannot_read_padding(self):
        p=LeafOwnershipPlan(64,256,4,4)
        with self.assertRaises(ValueError):p.resolve(64,live_batch=1)
        with self.assertRaises(ValueError):p.replicas_for_tile(16,live_batch=1)
        with self.assertRaises(ValueError):p.resolve(0,live_batch=5)
        # Slot capacity remains the full planned window, even in a tail.
        self.assertEqual(p.bank_elements,12*2*256)

    def test_no_tile_slot_reuse_before_stage_completion(self):
        p=LeafOwnershipPlan(256,64,4,4)
        owned=[p.resolve(r) for r in range(1024) if p.resolve(r).unit==0]
        self.assertEqual(len({x.leaf_slot for x in owned}),len(owned))
        self.assertEqual(p.resolve(0).leaf_slot,0)
        self.assertEqual(p.resolve(112).leaf_slot,4)

    def test_invalid_shapes_and_indices(self):
        for kwargs in (dict(group_leaves=3),dict(leaves_per_batch=7),dict(window_batch=0),
                       dict(group_leaves=True),dict(leaf_length=2.5),dict(bank_names=['a']),
                       dict(bank_names=('a','a')),dict(group_order='leaf-major')):
            args=dict(leaves_per_batch=64,leaf_length=256,group_leaves=4,window_batch=1)
            args.update(kwargs)
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):LeafOwnershipPlan(**args)
        p=LeafOwnershipPlan(64,256,2,1)
        for kwargs in (dict(replica=-1,element=0,bank='buf_a',plane=0),
                       dict(replica=0,element=256,bank='buf_a',plane=0),
                       dict(replica=0,element=0,bank='bad',plane=0),
                       dict(replica=0,element=0,bank='buf_a',plane=True)):
            with self.subTest(kwargs=kwargs),self.assertRaises(ValueError):p.resident_index(**kwargs)
        with self.assertRaises(ValueError):p.regular_jobs(7,4)
        with self.assertRaises(ValueError):p.physical_unit(0,28)
        with self.assertRaises(ValueError):p.tiles_for_unit(28)

    def test_immutable(self):
        p=LeafOwnershipPlan(64,256,2,1)
        with self.assertRaises(FrozenInstanceError):p.window_batch=4
        with self.assertRaises(FrozenInstanceError):p.resolve(0).leaf_slot=10


if __name__=='__main__':
    unittest.main()
