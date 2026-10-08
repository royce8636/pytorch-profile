# Owner(s): ["module: inductor"]
import os
import unittest

# Every copy stream can park on a wait; they must not share hardware queues.
os.environ.setdefault("CUDA_DEVICE_MAX_CONNECTIONS", "32")

import torch
from gpuco_torch.plan import parse_plan, Tier
from gpuco_torch.runtime.copy_list import CopyList
from gpuco_torch.runtime.executor import Executor
from gpuco_torch.runtime.flags import FlagBank
from gpuco_torch.runtime.registry import WeightRegistry
from torch.testing._internal.common_utils import run_tests, TestCase


D = 64
N = 6  # launches; launch i consumes group i
BIAS = 37  # an odd-sized member: the group's second member lands on a padded offset


def row(g, first=None, last=None, slot=None, **extra):
    r = {"group_id": g, "episode_id": g, "slot_id": g % 2 if slot is None else slot,
         "first_use": g if first is None else first, "last_use": g if last is None else last,
         "pos": 0, "source": "dram", "ring_off": -1, "issue_launch": -1, "issue_prev_iter": False}
    r.update(extra)
    return r


class TestCopyList(TestCase):
    def test_tenancy_is_last_use_rank_per_slot(self):
        cl = CopyList.from_sidecar([row(4), row(0), row(2), row(1), row(3)])
        self.assertEqual([(e.episode_id, e.slot_id, e.tenancy) for e in cl.entries],
                         [(0, 0, 1), (1, 1, 1), (2, 0, 2), (3, 1, 2), (4, 0, 3)])
        self.assertEqual(dict(cl.slot_tenants), {0: 3, 1: 2})
        self.assertEqual(cl.entries[0].source, Tier.DRAM)

    def test_enqueue_order_is_first_use(self):
        # g1 is used early but freed late: last-use rank puts it after g2.
        cl = CopyList.from_sidecar([row(0, 0, 0, slot=0), row(1, 1, 9, slot=1), row(2, 2, 3, slot=2)])
        self.assertEqual([e.group_id for e in cl.entries], [0, 2, 1])
        self.assertEqual([e.group_id for e in cl.enqueue_order], [0, 1, 2])

    def test_overlapping_tenants_are_infeasible(self):
        with self.assertRaisesRegex(ValueError, "overlap in executed order"):
            CopyList.from_sidecar([row(0, 0, 5, slot=0), row(1, 2, 3, slot=0)])

    def test_prefix_must_lead(self):
        cl = CopyList.from_sidecar([row(0, issue_launch=5, issue_prev_iter=True), row(1)])
        self.assertEqual([e.group_id for e in cl.prefix], [0])
        with self.assertRaisesRegex(ValueError, "must lead"):
            CopyList.from_sidecar([row(0), row(1, issue_launch=5, issue_prev_iter=True)])

    def test_stale_or_ambiguous_rows(self):
        bad = row(0)
        del bad["last_use"]
        with self.assertRaisesRegex(ValueError, "stale sidecar"):
            CopyList.from_sidecar([bad])
        with self.assertRaisesRegex(ValueError, "duplicate episode_id"):
            CopyList.from_sidecar([row(0), row(1, episode_id=0)])

    def test_episode_defaults_to_group(self):
        r = row(3)
        del r["episode_id"]
        self.assertEqual(CopyList.from_sidecar([r]).entries[0].episode_id, 3)


def plan_for(n_groups=N, slots=2):
    ops = []
    for g in range(n_groups):
        names = ["b1", "w1"] if g == 1 else [f"w{g}"]
        for name in names:
            ops.append({"type": "vram_prefetch_h2d", "tensor_name": name, "compiled_graph_input_name": name,
                        "compiled_graph_id": 0, "before_launch_id": g, "group_id": g, "slot_id": g % slots,
                        "tier": "dram", "source_pinned": True})
    return parse_plan({"summary": {"arena": {"slots": [{"slot_id": s, "class_mb": 0.1} for s in range(slots)]}},
                       "io_operations": ops})


class Model:
    """``N`` chained matmuls over streamed weights plus one resident weight,
    written against the instrumented-graph hooks as codegen would emit them."""

    def __init__(self, device, hooks=None, issue_at=None):
        g = torch.Generator(device="cpu").manual_seed(0)
        self.w = [torch.randn(D, D, generator=g).div_(8).to(device) for _ in range(N)]
        self.b1 = torch.randn(BIAS, generator=g).to(device)
        self.r = torch.randn(D, D, generator=g).div_(8).to(device)
        self.x = torch.randn(8, D, generator=g).to(device)
        self.hooks = hooks
        self.issue_at = issue_at or {}  # launch -> episodes whose ISSUE the graph writes after it
        self.skip_free = set()

    def names(self):
        return [("b1", self.b1)] + [(f"w{i}", w) for i, w in enumerate(self.w)] + [("r", self.r)]

    def __call__(self, scale=1.0):
        h = self.hooks
        if h is not None and h.begin_graph_iter(0) == 0:
            for name, t in self.names():
                h.register_weight_named(0, name, t)
        y = self.x
        for i in range(N):
            if h is not None:
                h.cudagraph_wait([i])
            y = torch.tanh(y @ self.w[i])
            if i == 1:
                y = y + torch.nn.functional.pad(self.b1, (0, D - BIAS))
            if h is not None:
                if i not in self.skip_free:
                    h.cudagraph_free([i])
                if i in self.issue_at:
                    h.cudagraph_issue(self.issue_at[i])
        return (y @ self.r) * scale


@unittest.skipIf(not torch.cuda.is_available(), "needs CUDA")
class TestExecutor(TestCase):
    def setUp(self):
        torch.backends.cuda.matmul.allow_tf32 = False
        self.device = torch.device("cuda", 0)

    def reference(self, scale=1.0):
        """The same kernels on never-streamed weights, captured the same way."""
        m = Model(self.device)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            m(scale)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            out = m(scale)
        g.replay()
        torch.cuda.synchronize()
        return out.clone()

    def streamed(self, rows=None, issue_at=None):
        plan = plan_for()
        reg = WeightRegistry(plan, self.device)
        m = Model(self.device, reg, issue_at)
        m()  # registration block: takes host copies and repoints the streamed weights into the arena
        torch.cuda.synchronize()
        ex = Executor(reg, CopyList.from_sidecar(rows or [row(g) for g in range(N)]), self.device)
        return m, reg, ex

    def run_steps(self, ex, n, expect, kind=None):
        for _ in range(n):
            p = ex.step(kind)
            torch.cuda.synchronize()
            self.assertEqual(ex.static_output(p, kind), expect, atol=0, rtol=0)

    def test_streamed_replay_is_byte_exact(self):
        ref = self.reference()
        m, reg, ex = self.streamed()
        self.assertEqual(reg.member_offset[id(m.b1)], (0, BIAS * 4))
        self.assertEqual(reg.member_offset[id(m.w[1])], (512, D * D * 4))  # padded past the bias
        self.assertEqual(m.w[0].untyped_storage().data_ptr(), reg.arena.storage.data_ptr())
        self.assertNotEqual(m.r.untyped_storage().data_ptr(), reg.arena.storage.data_ptr())  # resident
        ex.capture(m)
        self.run_steps(ex, 7, ref)  # 2 slots x 3 tenants: tenant 1 waits on the previous iteration's last
        rep = ex.report()
        self.assertEqual((rep["iterations"], rep["n_slots"], rep["max_tenants_per_slot"]), (7, 2, 3))

    def test_output_depends_on_the_copies(self):
        ref = self.reference()
        m, reg, ex = self.streamed()
        ex.capture(m)
        self.run_steps(ex, 2, ref)
        block = ex.engine.dram.blocks[3]
        saved = block.clone()
        block.zero_()
        p = ex.step()
        torch.cuda.synchronize()
        self.assertNotEqual(ex.static_output(p), ref)
        block.copy_(saved)
        self.run_steps(ex, 2, ref)

    def test_issue_gates(self):
        ref = self.reference()
        # Episode 5's copy waits for the graph's ISSUE after launch 3; episode 0's
        # writer is the previous iteration's graph (after launch 5), preset at start.
        rows = [row(g) for g in range(N)]
        rows[5].update(issue_launch=3)
        rows[0].update(issue_launch=5, issue_prev_iter=True)
        m, reg, ex = self.streamed(rows, issue_at={3: [5], 5: [0]})
        ex.capture(m)
        self.assertEqual(ex.flags.issue.tolist()[:N], [1, 0, 0, 0, 0, 0])
        self.run_steps(ex, 5, ref)

    def test_second_callable_over_the_same_copy_list(self):
        ref, ref2 = self.reference(), self.reference(2.0)
        m, reg, ex = self.streamed()
        ex.capture(m)
        ex.capture(lambda: m(2.0), kind="aux", n_warm=2)
        for _ in range(3):
            self.run_steps(ex, 1, ref)
            self.run_steps(ex, 1, ref2, kind="aux")

    def test_compute_only_then_streamed(self):
        ref = self.reference()
        m, reg, ex = self.streamed()
        ex.capture(m)
        self.run_steps(ex, 2, ref)
        for _ in range(3):
            ex.step(compute_only=True)
        torch.cuda.synchronize()
        self.run_steps(ex, 3, ref)

    def test_missing_free_writer_fails_at_capture(self):
        m, reg, ex = self.streamed()
        m.skip_free = {4}
        with self.assertRaisesRegex(RuntimeError, r"missing \[4\]"):
            ex.capture(m)

    def test_registered_group_without_copy_entry(self):
        with self.assertRaisesRegex(RuntimeError, r"groups \[5\] are registered"):
            self.streamed([row(g) for g in range(N - 1)])

    def test_init_pattern_is_history_independent(self):
        fb = FlagBank(4, {0: 3, 1: 1}, self.device, {2})
        fb.init_flags()
        first = (fb.ready.tolist(), fb.freed.tolist(), fb.issue.tolist())
        fb.ready.fill_(1)
        fb.freed.fill_(1)
        fb.issue.zero_()
        fb.init_flags()
        self.assertEqual((fb.ready.tolist(), fb.freed.tolist(), fb.issue.tolist()), first)
        self.assertEqual(first, ([0, 0, 0, 0], [0, 0, 1, 1], [0, 0, 1, 0]))
        self.assertEqual((fb.pred(0, 1), fb.pred(0, 3), fb.pred(1, 1)), (3, 2, 1))
        with self.assertRaises(IndexError):
            fb.freed_index(0, 4)


@unittest.skipIf(not torch.cuda.is_available(), "needs CUDA")
class TestRegistry(TestCase):
    def test_plan_members_stream_and_others_stay(self):
        dev = torch.device("cuda", 0)
        reg = WeightRegistry(plan_for(), dev)
        w0, other = torch.full((D, D), 3.0, device=dev), torch.ones(D, D, device=dev)
        reg.register_weight_named(0, "w0", w0)  # in the plan: host copy taken, repointed
        self.assertEqual(w0.untyped_storage().data_ptr(), reg.arena.storage.data_ptr())
        self.assertEqual(reg.host_bytes(w0).view(torch.float32)[0].item(), 3.0)
        reg.register_weight_named(0, "zzz", other)  # not in the plan: resident
        self.assertNotEqual(other.untyped_storage().data_ptr(), reg.arena.storage.data_ptr())
        self.assertEqual(reg.group_members, {0: [w0]})

    def test_views_and_oversized_groups_are_rejected(self):
        dev = torch.device("cuda", 0)
        reg = WeightRegistry(plan_for(), dev)
        with self.assertRaisesRegex(ValueError, "whole storage"):
            reg.adopt(torch.ones(2 * D, D, device=dev)[D:])
        big = torch.ones(200, 200, device=dev)  # 160 KB > the 100 KB slot
        with self.assertRaisesRegex(RuntimeError, "needs 160000 B in slot 0"):
            reg.register_weight_named(0, "w0", big)


if __name__ == "__main__":
    run_tests()
