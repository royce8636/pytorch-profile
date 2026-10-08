# Owner(s): ["module: inductor"]
import copy
import json
import os
import tempfile

from gpuco_torch.plan import (
    load_crosswalk,
    load_plan,
    parse_plan,
    PlanError,
    Tier,
    tiers_by_checkpoint,
)
from torch.testing._internal.common_utils import (
    instantiate_parametrized_tests,
    parametrize,
    run_tests,
    TestCase,
)


def reload(name, group, slot, anchor, tier="ssd", graph=0, **extra):
    op = {
        "type": "vram_prefetch_h2d",
        "tensor_name": name,
        "compiled_graph_input_name": name,
        "compiled_graph_id": graph,
        "before_launch_id": anchor,
        "group_id": group,
        "slot_id": slot,
        "tier": tier,
        "source_pinned": tier == "dram",
        "iter_mask": [],
        "issue_compiled_graph_id": -1,
    }
    if tier == "ssd":
        op["ring_off"] = 4096 * group
    op.update(extra)
    return op


def plan_dict():
    """Today's scheduler format: three groups over two slots, plus the
    planner-only content the executor ignores."""
    return {
        "summary": {
            "graph_multiplicity": {"0": 1},
            "cross_graph_evict": False,
            "arena": {
                "slots": [{"slot_id": 1, "class_mb": 2.5}, {"slot_id": 0, "class_mb": 1}],
                "ring_bytes": 1 << 20,
                "summary": {"n_slots": 2},
            },
            "modeled_peak_mb": 3.5,
        },
        "nodes": [{"idx": 0, "name": "mm", "resource_kind": "GPU"}],
        "io_operations": [
            reload("arg1_1", 0, 0, 3, tier="dram"),
            {"type": "prefetch", "tensor_name": "arg2_1", "before_node": 1},
            reload("arg2_1", 1, 1, 7),
            reload("arg3_1", 1, 1, 9),
            reload("arg4_1", 2, 0, 12),
            reload("arg4_1", 2, 0, 5),
            {"type": "vram_evict_d2h", "tensor_name": "arg1_1", "after_node": 4},
        ],
        "spill_decisions": [],
        "cold_start_prefetches": [{"tensor_name": "arg9_1", "attach_before_node": 0}],
        "steady_state_resident": [],
        "compilation_hash": "abc123",
    }


class TestPlanLoad(TestCase):
    def test_groups_slots_and_tiers(self):
        p = parse_plan(plan_dict())
        self.assertEqual(sorted(p.groups), [0, 1, 2])
        self.assertEqual({g.group_id: g.tier for g in p.groups.values()},
                         {0: Tier.DRAM, 1: Tier.SSD, 2: Tier.SSD})
        self.assertEqual([m.name for m in p.groups[1].members], ["arg2_1", "arg3_1"])
        self.assertEqual(p.groups[1].ring_off, 4096)
        self.assertEqual(p.groups[0].ring_off, -1)
        self.assertEqual(p.slots[1].capacity, 2_500_000)
        self.assertEqual(dict(p.slot_tenants), {0: (0, 2), 1: (1,)})
        self.assertEqual((p.ring_bytes, p.compilation_hash, dict(p.graph_multiplicity), p.cross_graph_evict),
                         (1 << 20, "abc123", {0: 1}, False))

    def test_reloads_of_one_member_become_sorted_anchors(self):
        (m,) = parse_plan(plan_dict()).groups[2].members
        self.assertEqual(m.anchors, (5, 12))

    def test_lookup_by_graph_input(self):
        p = parse_plan(plan_dict())
        self.assertEqual(p.group_of(0, "arg3_1").group_id, 1)
        self.assertIsNone(p.group_of(0, "arg9_1"))  # cold start only: resident
        self.assertIsNone(p.group_of(1, "arg3_1"))
        self.assertEqual([g.group_id for g in p.groups_in_graph(0)], [0, 1, 2])
        self.assertEqual([g.group_id for g in p.groups_on(Tier.SSD)], [1, 2])

    def test_any_graph_member_matches_every_graph(self):
        d = plan_dict()
        d["io_operations"][0]["compiled_graph_id"] = -1
        p = parse_plan(d)
        self.assertEqual(p.group_of(3, "arg1_1").group_id, 0)
        self.assertEqual([g.group_id for g in p.groups_in_graph(3)], [0])

    def test_slot_layout_is_aligned_prefix_sum_in_slot_order(self):
        layout, total = parse_plan(plan_dict()).slot_layout(512)
        self.assertEqual(layout, {0: (0, 1_000_000), 1: (1_000_448, 2_500_000)})
        self.assertEqual(total, 1_000_448 + 2_500_096)

    def test_untagged_plan_is_dram_only(self):
        d = plan_dict()
        for op in d["io_operations"]:
            op.pop("tier", None)
            op["source_pinned"] = False
        self.assertEqual({g.tier for g in parse_plan(d).groups.values()}, {Tier.DRAM})

    def test_explicit_format_version(self):
        d = plan_dict()
        d["format_version"] = 1
        parse_plan(d)
        d["format_version"] = 2
        with self.assertRaisesRegex(PlanError, "format_version 2"):
            parse_plan(d)

    def test_load_plan_from_file(self):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump(plan_dict(), f)
        try:
            p = load_plan(f.name)
            self.assertEqual(p.source, f.name)
            self.assertEqual(len(p.groups), 3)
        finally:
            os.unlink(f.name)


def _set(op_index, **fields):
    def edit(d):
        d["io_operations"][op_index].update(fields)
    return edit


def _drop(op_index, key):
    def edit(d):
        d["io_operations"][op_index].pop(key)
    return edit


INVALID = {
    "straddles_tiers": (_set(3, tier="dram", source_pinned=True), "straddles tiers"),
    "spans_slots": (_set(3, slot_id=0), "spans slots 1 and 0"),
    "ring_offsets_differ": (_set(3, ring_off=0), "ring offsets"),
    "unknown_slot": (_set(0, slot_id=7), "slot 7 is not in summary.arena.slots"),
    "unannotated": (_set(0, group_id=-1), "unannotated reload"),
    "negative_anchor": (_set(0, before_launch_id=-1), "before_launch_id -1"),
    "no_input_name": (_set(0, compiled_graph_input_name=""), "empty compiled_graph_input_name"),
    "missing_group_id": (_drop(0, "group_id"), "group_id"),
    "bad_tier": (_set(0, tier="nvme"), "tier 'nvme'"),
    "tier_vs_pinned": (_set(2, source_pinned=True), "contradicts source_pinned"),
    "iter_mask": (_set(0, iter_mask=[0]), "iter_mask"),
    "cross_graph_issue": (_set(0, issue_compiled_graph_id=4), "cross-graph issue"),
    "two_groups": (_set(2, compiled_graph_input_name="arg1_1", group_id=1), "in groups 0 and 1"),
    "unknown_op": (lambda d: d["io_operations"].append({"type": "vram_swap"}), "unknown operation type"),
    "spill": (lambda d: d["spill_decisions"].append({"tensor_name": "x", "evict_after_node": 1}), "spill_decisions"),
    "no_slot_table": (lambda d: d["summary"]["arena"].update(slots=[]), "slot table"),
    "zero_capacity": (lambda d: d["summary"]["arena"]["slots"].append({"slot_id": 5, "class_mb": 0}), "capacity 0"),
    "duplicate_slot": (lambda d: d["summary"]["arena"]["slots"].append({"slot_id": 0, "class_mb": 1}), "listed twice"),
    "not_a_plan": (lambda d: d.pop("io_operations"), "not a plan"),
}


class TestPlanValidation(TestCase):
    @parametrize("case", list(INVALID))
    def test_rejects(self, case):
        edit, message = INVALID[case]
        d = plan_dict()
        edit(d)
        with self.assertRaisesRegex(PlanError, message):
            parse_plan(d)

    def test_reports_every_problem(self):
        d = plan_dict()
        for op in d["io_operations"]:
            if op["type"] == "vram_prefetch_h2d":
                op["slot_id"] = 9
        with self.assertRaises(PlanError) as cm:
            parse_plan(d)
        self.assertEqual(sum("slot 9 is not in" in p for p in cm.exception.problems), 5)

    def test_input_is_not_modified(self):
        d = plan_dict()
        before = copy.deepcopy(d)
        parse_plan(d)
        self.assertEqual(d, before)


class TestCrosswalk(TestCase):
    def _crosswalk(self, graphs, unmatched=()):
        with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
            json.dump({"graphs": graphs, "unmatched": list(unmatched)}, f)
        self.addCleanup(os.unlink, f.name)
        return load_crosswalk(f.name)

    def _names(self):
        return {"arg1_1": "embed.weight", "arg2_1": "l0.q.weight", "arg3_1": "l0.k.weight",
                "arg4_1": "lm_head.weight", "arg5_1": "l0.norm.weight"}

    def test_tiers_by_checkpoint(self):
        xw = self._crosswalk({"0": self._names()}, unmatched=[[0, "arg7_1", [2, 8]]])
        self.assertEqual(xw.checkpoint_name(0, "arg2_1"), "l0.q.weight")
        self.assertEqual(xw.unmatched, ((0, "arg7_1"),))
        self.assertEqual(tiers_by_checkpoint(parse_plan(plan_dict()), xw), {
            "embed.weight": Tier.DRAM, "l0.q.weight": Tier.SSD, "l0.k.weight": Tier.SSD,
            "lm_head.weight": Tier.SSD,
        })

    def test_streamed_input_without_checkpoint_name(self):
        names = self._names()
        del names["arg3_1"]
        with self.assertRaisesRegex(PlanError, "arg3_1 .* has no checkpoint name"):
            tiers_by_checkpoint(parse_plan(plan_dict()), self._crosswalk({"0": names}))

    def test_tied_names_are_rejected(self):
        names = self._names()
        names["arg4_1"] = "embed.weight"
        with self.assertRaisesRegex(PlanError, "tied weights"):
            self._crosswalk({"0": names})

    def test_one_checkpoint_tensor_on_two_tiers(self):
        xw = self._crosswalk({"0": self._names(), "1": {"arg1_1": "l0.q.weight"}})
        d = plan_dict()
        d["io_operations"][0]["compiled_graph_id"] = 1
        with self.assertRaisesRegex(PlanError, "homed on both"):
            tiers_by_checkpoint(parse_plan(d), xw)


instantiate_parametrized_tests(TestPlanValidation)


if __name__ == "__main__":
    run_tests()
