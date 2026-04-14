# Owner(s): ["module: inductor"]
import json
import os
import tempfile

import torch
from torch._inductor.codegen.wrapper import (
    DeviceSynchronizeLine,
    EventRecordLine,
    EventSynchronizeLine,
    KernelCallLine,
    PythonWrapperCodegen,
    StreamSynchronizeLine,
    SyncMarkerLine,
    WaitEventLine,
)
from torch._inductor.test_case import run_tests, TestCase
from torch._inductor.virtualized import V
from torch._inductor.weight_streaming.codegen import (
    _op_to_call,
    _vram_op_to_call,
    ScheduleAdapter,
)
from torch._inductor.weight_streaming.plan import (
    ColdStartOp,
    EvictDramOp,
    EvictVramOp,
    H2DPrefetchOp,
    IOSchedule,
    load_io_schedule,
    PrefetchOp,
    TensorEntry,
)
from torch._inductor.weight_streaming.runtime import WeightStreamRuntime
from torch.testing._internal.inductor_utils import HAS_GPU


class TestIOSchedule(TestCase):
    def test_load_io_schedule_from_scheduler_format(self):
        """Test loading the actual scheduler output format."""
        schedule_data = {
            "summary": {"io_model": "test"},
            "nodes": [
                {"idx": 0, "name": "aten::empty", "resource_kind": "cpu_thread"},
                {"idx": 1, "name": "triton_fused_mm_0", "resource_kind": "gpu_stream"},
            ],
            "io_operations": [
                {
                    "type": "prefetch",
                    "tensor_name": "param_0_1",
                    "tensor_kind": "WEIGHT",
                    "before_node": 1,
                    "after_node": -1,
                    "eager_start": True,
                },
                {
                    "type": "vram_prefetch_h2d",
                    "tensor_name": "param_0_1",
                    "tensor_kind": "WEIGHT",
                    "before_node": 1,
                },
                {
                    "type": "vram_evict_d2h",
                    "tensor_name": "param_0_1",
                    "tensor_kind": "WEIGHT",
                    "after_node": 1,
                },
            ],
            "spill_decisions": [
                {
                    "tensor_name": "param_0_1",
                    "tensor_kind": "WEIGHT",
                    "evict_after_node": 1,
                    "reason": "immediate_evict_after_use",
                },
            ],
            "cold_start_prefetches": [
                {"tensor_name": "param_cold", "attach_before_node": 0},
            ],
            "tensor_metadata": {
                "param_0_1": {
                    "kind": "WEIGHT",
                    "size_bytes": 16384,
                    "dtype": "float16",
                    "shape": [64, 64],
                }
            },
        }

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(schedule_data, f)
            f.flush()
            path = f.name

        try:
            schedule = load_io_schedule(path)
            self.assertIsInstance(schedule, IOSchedule)
            self.assertEqual(len(schedule.nodes), 2)

            # SSD prefetches
            self.assertEqual(len(schedule.prefetches), 1)
            self.assertEqual(schedule.prefetches[0].tensor_name, "param_0_1")
            self.assertTrue(schedule.prefetches[0].eager_start)

            # H2D prefetches
            self.assertEqual(len(schedule.h2d_prefetches), 1)
            self.assertEqual(schedule.h2d_prefetches[0].before_node, 1)

            # VRAM evictions
            self.assertEqual(len(schedule.evict_vram), 1)
            self.assertEqual(schedule.evict_vram[0].after_node, 1)

            # DRAM evictions (from spill_decisions)
            self.assertEqual(len(schedule.evict_dram), 1)
            self.assertEqual(schedule.evict_dram[0].evict_after_node, 1)

            # Cold starts
            self.assertEqual(len(schedule.cold_starts), 1)

            # Tensor metadata
            self.assertIn("param_0_1", schedule.tensors)
            self.assertEqual(schedule.tensors["param_0_1"].dtype, "float16")
        finally:
            os.unlink(path)

    def test_load_io_schedule_with_csv_enrichment(self):
        schedule_data = {
            "nodes": [
                {"idx": 0, "name": "aten::empty"},
                {"idx": 1, "name": "triton_fused_mm_0"},
            ],
            "io_operations": [],
            "spill_decisions": [],
            "cold_start_prefetches": [],
        }

        csv_content = (
            "idx,name,resource_kind\n"
            "0,aten::empty,cpu_thread\n"
            "1,triton_fused_mm_0,gpu_stream\n"
        )

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(schedule_data, f)
            f.flush()
            json_path = f.name

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".csv", delete=False
        ) as f:
            f.write(csv_content)
            f.flush()
            csv_path = f.name

        try:
            schedule = load_io_schedule(json_path, nodes_csv=csv_path)
            self.assertEqual(schedule.nodes[0]["resource_kind"], "cpu_thread")
            self.assertEqual(schedule.nodes[1]["resource_kind"], "gpu_stream")
        finally:
            os.unlink(json_path)
            os.unlink(csv_path)

    def test_dataclasses(self):
        entry = TensorEntry(
            name="w", kind="WEIGHT", size_bytes=100, dtype="float32", shape=[10, 10]
        )
        self.assertEqual(entry.name, "w")

        prefetch = PrefetchOp(tensor_name="w", before_node=5, after_node=3)
        self.assertEqual(prefetch.before_node, 5)

        h2d = H2DPrefetchOp(tensor_name="w", before_node=5)
        self.assertEqual(h2d.before_node, 5)

        evict_vram = EvictVramOp(tensor_name="w", after_node=7)
        self.assertEqual(evict_vram.after_node, 7)

        evict_dram = EvictDramOp(tensor_name="w", evict_after_node=7)
        self.assertEqual(evict_dram.evict_after_node, 7)

        cold = ColdStartOp(tensor_name="w", attach_before_node=0)
        self.assertEqual(cold.attach_before_node, 0)


class TestScheduleAdapter(TestCase):
    def _make_nodes(self, *specs):
        """Helper: specs are (idx, name, resource_kind) tuples."""
        return [
            {"idx": idx, "name": name, "resource_kind": rk}
            for idx, name, rk in specs
        ]

    def test_gpu_cpu_classification(self):
        nodes = self._make_nodes(
            (0, "aten::empty", "cpu_thread"),
            (1, "triton_fused_0", "gpu_stream"),
            (2, "aten::add", "cpu_thread"),
            (3, "triton_fused_1", "gpu_stream"),
        )
        schedule = IOSchedule(nodes=nodes)
        adapter = ScheduleAdapter(schedule)
        self.assertEqual(adapter._gpu_trace_indices, [1, 3])
        self.assertEqual(adapter._trace_to_wrapper, {1: 0, 3: 1})

    def test_ssd_prefetch_on_gpu_node(self):
        nodes = self._make_nodes(
            (0, "aten::empty", "cpu_thread"),
            (1, "triton_fused_0", "gpu_stream"),
            (2, "triton_fused_1", "gpu_stream"),
        )
        prefetches = [PrefetchOp(tensor_name="w", before_node=2, after_node=0)]
        schedule = IOSchedule(nodes=nodes, prefetches=prefetches)
        adapter = ScheduleAdapter(schedule)
        # before_node=2 is GPU, maps to wrapper_idx=1
        self.assertIn(1, adapter.before_kernel)
        self.assertIsInstance(adapter.before_kernel[1][0], PrefetchOp)

    def test_h2d_prefetch_mapped(self):
        nodes = self._make_nodes(
            (0, "triton_0", "gpu_stream"),
            (1, "triton_1", "gpu_stream"),
        )
        h2d_ops = [H2DPrefetchOp(tensor_name="w", before_node=1)]
        schedule = IOSchedule(nodes=nodes, h2d_prefetches=h2d_ops)
        adapter = ScheduleAdapter(schedule)
        self.assertIn(1, adapter.before_kernel)
        self.assertIsInstance(adapter.before_kernel[1][0], H2DPrefetchOp)

    def test_evict_vram_mapped(self):
        nodes = self._make_nodes(
            (0, "triton_0", "gpu_stream"),
            (1, "triton_1", "gpu_stream"),
        )
        evict_ops = [EvictVramOp(tensor_name="w", after_node=0)]
        schedule = IOSchedule(nodes=nodes, evict_vram=evict_ops)
        adapter = ScheduleAdapter(schedule)
        self.assertIn(0, adapter.after_kernel)
        self.assertIsInstance(adapter.after_kernel[0][0], EvictVramOp)

    def test_evict_dram_mapped(self):
        nodes = self._make_nodes(
            (0, "triton_0", "gpu_stream"),
            (1, "triton_1", "gpu_stream"),
        )
        evict_ops = [EvictDramOp(tensor_name="w", evict_after_node=1)]
        schedule = IOSchedule(nodes=nodes, evict_dram=evict_ops)
        adapter = ScheduleAdapter(schedule)
        self.assertIn(1, adapter.after_kernel)
        self.assertIsInstance(adapter.after_kernel[1][0], EvictDramOp)

    def test_prefetch_on_cpu_node_snaps_forward(self):
        nodes = self._make_nodes(
            (0, "aten::empty", "cpu_thread"),
            (1, "aten::add", "cpu_thread"),
            (2, "triton_fused_0", "gpu_stream"),
        )
        prefetches = [PrefetchOp(tensor_name="w", before_node=1, after_node=-1)]
        schedule = IOSchedule(nodes=nodes, prefetches=prefetches)
        adapter = ScheduleAdapter(schedule)
        self.assertIn(0, adapter.before_kernel)

    def test_evict_on_cpu_node_snaps_backward(self):
        nodes = self._make_nodes(
            (0, "triton_fused_0", "gpu_stream"),
            (1, "aten::empty", "cpu_thread"),
            (2, "triton_fused_1", "gpu_stream"),
        )
        evict_ops = [EvictDramOp(tensor_name="w", evict_after_node=1)]
        schedule = IOSchedule(nodes=nodes, evict_dram=evict_ops)
        adapter = ScheduleAdapter(schedule)
        self.assertIn(0, adapter.after_kernel)

    def test_cold_start_ops(self):
        nodes = self._make_nodes((0, "triton_0", "gpu_stream"))
        cold_starts = [ColdStartOp(tensor_name="w", attach_before_node=0)]
        schedule = IOSchedule(nodes=nodes, cold_starts=cold_starts)
        adapter = ScheduleAdapter(schedule)
        self.assertEqual(len(adapter.startup_ops), 1)

    def test_vram_and_dram_evicts_both_mapped(self):
        """Both VRAM and DRAM evictions on the same kernel are preserved."""
        nodes = self._make_nodes(
            (0, "triton_0", "gpu_stream"),
            (1, "triton_1", "gpu_stream"),
        )
        schedule = IOSchedule(
            nodes=nodes,
            evict_vram=[EvictVramOp(tensor_name="w", after_node=0)],
            evict_dram=[EvictDramOp(tensor_name="w", evict_after_node=0)],
        )
        adapter = ScheduleAdapter(schedule)
        post_ops = adapter.after_kernel[0]
        self.assertEqual(len(post_ops), 2)
        types = {type(op) for op in post_ops}
        self.assertEqual(types, {EvictVramOp, EvictDramOp})


class TestOpToCall(TestCase):
    def test_prefetch(self):
        op = PrefetchOp(tensor_name="w", before_node=0, after_node=-1)
        self.assertEqual(_op_to_call(op), "_ws_rt.ssd_prefetch('w')")

    def test_h2d_prefetch(self):
        op = H2DPrefetchOp(tensor_name="w", before_node=0)
        self.assertEqual(_op_to_call(op), "_ws_rt.h2d_prefetch('w')")

    def test_evict_vram(self):
        op = EvictVramOp(tensor_name="w", after_node=0)
        self.assertEqual(_op_to_call(op), "_ws_rt.evict_vram('w')")

    def test_evict_dram(self):
        op = EvictDramOp(tensor_name="w", evict_after_node=0)
        self.assertEqual(_op_to_call(op), "_ws_rt.evict_dram('w')")


class TestMixedLaunchIdSchedule(TestCase):
    """Schedules mixing ops with and without launch IDs."""

    def _make_nodes(self, *specs):
        return [
            {"idx": idx, "name": name, "resource_kind": rk}
            for idx, name, rk in specs
        ]

    def test_exact_and_legacy_ops_both_served(self):
        """Ops with launch IDs use exact path; ops without use rescaling."""
        nodes = self._make_nodes(
            (0, "triton_0", "gpu_stream"),
            (1, "triton_1", "gpu_stream"),
        )
        schedule = IOSchedule(
            nodes=nodes,
            prefetches=[
                PrefetchOp("exact", before_node=0, after_node=-1, before_launch_id=5),
                PrefetchOp("legacy", before_node=1, after_node=-1),
            ],
        )
        adapter = ScheduleAdapter(schedule)

        # Exact path serves the op with launch_id=5
        bl = adapter.before_launch
        exact_names = {op.tensor_name for ops in bl.values() for op in ops}
        self.assertIn("exact", exact_names)
        self.assertNotIn("legacy", exact_names)

        # Rescaled path serves the op without launch_id
        rescaled = adapter.rescale_before_kernel(2)
        legacy_names = {
            op.tensor_name for ops in rescaled.values() for op in ops
        }
        self.assertIn("legacy", legacy_names)
        self.assertNotIn("exact", legacy_names)

    def test_post_ops_exact_and_legacy_split(self):
        """Post-kernel ops: exact EvictDram + legacy EvictDram."""
        nodes = self._make_nodes(
            (0, "triton_0", "gpu_stream"),
            (1, "triton_1", "gpu_stream"),
        )
        schedule = IOSchedule(
            nodes=nodes,
            evict_dram=[
                EvictDramOp("exact_d", evict_after_node=0, after_launch_id=10),
                EvictDramOp("legacy_d", evict_after_node=1),
            ],
        )
        adapter = ScheduleAdapter(schedule)

        al = adapter.after_launch
        exact_names = {op.tensor_name for ops in al.values() for op in ops}
        self.assertEqual(exact_names, {"exact_d"})

        rescaled = adapter.rescale_after_kernel(2)
        legacy_names = {
            op.tensor_name for ops in rescaled.values() for op in ops
        }
        self.assertEqual(legacy_names, {"legacy_d"})


class TestVramOpResolution(TestCase):
    """VRAM ops resolved via compiled_tensor_id to variable-ref calls."""

    def test_h2d_prefetch_emits_variable_ref(self):
        tid_map = {0: "arg0_1", 1: "arg1_1"}
        op = H2DPrefetchOp("w", before_node=1, compiled_tensor_id=0)
        result = _vram_op_to_call(op, tid_map)
        self.assertEqual(result, "_ws_rt.h2d_prefetch(arg0_1)")

    def test_evict_vram_emits_variable_ref(self):
        tid_map = {0: "arg0_1", 1: "arg1_1"}
        op = EvictVramOp("w", after_node=1, compiled_tensor_id=1)
        result = _vram_op_to_call(op, tid_map)
        self.assertEqual(result, "_ws_rt.evict_vram(arg1_1)")

    def test_ssd_dram_ops_still_use_string_names(self):
        ssd = PrefetchOp("w", before_node=0, after_node=-1)
        self.assertEqual(_op_to_call(ssd), "_ws_rt.ssd_prefetch('w')")
        dram = EvictDramOp("w", evict_after_node=0)
        self.assertEqual(_op_to_call(dram), "_ws_rt.evict_dram('w')")

    def test_missing_compiled_tensor_id_raises(self):
        tid_map = {0: "arg0_1"}
        op = H2DPrefetchOp("w", before_node=1, compiled_tensor_id=-1)
        with self.assertRaises(ValueError, msg="missing compiled_tensor_id"):
            _vram_op_to_call(op, tid_map)

    def test_invalid_compiled_tensor_id_raises(self):
        tid_map = {0: "arg0_1"}
        op = EvictVramOp("w", after_node=1, compiled_tensor_id=9999)
        with self.assertRaises(ValueError, msg="not found in graph inputs"):
            _vram_op_to_call(op, tid_map)

    def test_vram_ops_in_exact_launch_maps(self):
        """VRAM ops with launch IDs appear in before_launch/after_launch."""
        schedule = IOSchedule(
            h2d_prefetches=[
                H2DPrefetchOp("w", before_node=1, before_launch_id=7,
                              compiled_tensor_id=0),
            ],
            evict_vram=[
                EvictVramOp("w", after_node=1, after_launch_id=7,
                            compiled_tensor_id=0),
            ],
            evict_dram=[
                EvictDramOp("d", evict_after_node=1, after_launch_id=7),
            ],
        )
        adapter = ScheduleAdapter(schedule)
        bl = adapter.before_launch
        al = adapter.after_launch
        all_before = [op for ops in bl.values() for op in ops]
        all_after = [op for ops in al.values() for op in ops]
        self.assertTrue(any(isinstance(op, H2DPrefetchOp) for op in all_before))
        self.assertTrue(any(isinstance(op, EvictVramOp) for op in all_after))
        self.assertTrue(any(isinstance(op, EvictDramOp) for op in all_after))


class TestLaunchIdFields(TestCase):
    """Verify launch_id fields are parsed from JSON."""

    def test_prefetch_launch_id_parsed(self):
        data = {
            "nodes": [],
            "io_operations": [
                {
                    "type": "prefetch",
                    "tensor_name": "w",
                    "before_node": 1,
                    "after_node": -1,
                    "before_launch_id": 42,
                },
            ],
            "spill_decisions": [],
            "cold_start_prefetches": [],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            schedule = load_io_schedule(path)
            self.assertEqual(schedule.prefetches[0].before_launch_id, 42)
        finally:
            os.unlink(path)

    def test_evict_dram_launch_id_parsed(self):
        data = {
            "nodes": [],
            "io_operations": [],
            "spill_decisions": [
                {
                    "tensor_name": "w",
                    "evict_after_node": 5,
                    "after_launch_id": 99,
                },
            ],
            "cold_start_prefetches": [],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            schedule = load_io_schedule(path)
            self.assertEqual(schedule.evict_dram[0].after_launch_id, 99)
        finally:
            os.unlink(path)

    def test_compilation_hash_parsed(self):
        data = {
            "nodes": [],
            "io_operations": [],
            "spill_decisions": [],
            "cold_start_prefetches": [],
            "compilation_hash": "abc123def456",
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            schedule = load_io_schedule(path)
            self.assertEqual(schedule.compilation_hash, "abc123def456")
        finally:
            os.unlink(path)

    def test_missing_launch_id_defaults_to_negative(self):
        """Ops without launch_id fields default to -1."""
        data = {
            "nodes": [],
            "io_operations": [
                {"type": "prefetch", "tensor_name": "w", "before_node": 0},
                {"type": "vram_evict_d2h", "tensor_name": "w", "after_node": 0},
            ],
            "spill_decisions": [
                {"tensor_name": "w", "evict_after_node": 0},
            ],
            "cold_start_prefetches": [
                {"tensor_name": "w", "attach_before_node": 0},
            ],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            schedule = load_io_schedule(path)
            self.assertEqual(schedule.prefetches[0].before_launch_id, -1)
            self.assertEqual(schedule.evict_vram[0].after_launch_id, -1)
            self.assertEqual(schedule.evict_dram[0].after_launch_id, -1)
            self.assertEqual(schedule.cold_starts[0].before_launch_id, -1)
        finally:
            os.unlink(path)

    def test_compiled_tensor_id_parsed(self):
        data = {
            "nodes": [],
            "io_operations": [
                {
                    "type": "vram_prefetch_h2d",
                    "tensor_name": "w",
                    "before_node": 1,
                    "before_launch_id": 5,
                    "compiled_tensor_id": 3,
                },
                {
                    "type": "vram_evict_d2h",
                    "tensor_name": "w",
                    "after_node": 1,
                    "after_launch_id": 5,
                    "compiled_tensor_id": 3,
                },
            ],
            "spill_decisions": [],
            "cold_start_prefetches": [],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            schedule = load_io_schedule(path)
            self.assertEqual(schedule.h2d_prefetches[0].compiled_tensor_id, 3)
            self.assertEqual(schedule.evict_vram[0].compiled_tensor_id, 3)
        finally:
            os.unlink(path)

    def test_missing_compiled_tensor_id_defaults_to_negative(self):
        data = {
            "nodes": [],
            "io_operations": [
                {"type": "vram_prefetch_h2d", "tensor_name": "w", "before_node": 1},
                {"type": "vram_evict_d2h", "tensor_name": "w", "after_node": 1},
            ],
            "spill_decisions": [],
            "cold_start_prefetches": [],
        }
        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".json", delete=False
        ) as f:
            json.dump(data, f)
            f.flush()
            path = f.name
        try:
            schedule = load_io_schedule(path)
            self.assertEqual(schedule.h2d_prefetches[0].compiled_tensor_id, -1)
            self.assertEqual(schedule.evict_vram[0].compiled_tensor_id, -1)
        finally:
            os.unlink(path)


class TestSyncLineCodegen(TestCase):
    """Tests for typed sync WrapperLines and SyncMarkerLine codegen."""

    def _codegen_to_str(self, line):
        from torch._inductor.utils import IndentedBuffer

        buf = IndentedBuffer()
        line.codegen(buf)
        return buf.getvalue()

    def test_stream_synchronize_line_codegen(self):
        line = StreamSynchronizeLine(wrapper=None, stream_idx=1)
        output = self._codegen_to_str(line)
        self.assertIn(".synchronize()", output)

    def test_event_record_line_codegen(self):
        line = EventRecordLine(wrapper=None, event_var="_ev0", stream_idx=0)
        output = self._codegen_to_str(line)
        self.assertIn("torch.cuda.Event()", output)
        self.assertIn("_ev0", output)
        self.assertIn(".record(", output)

    def test_wait_event_line_codegen(self):
        line = WaitEventLine(wrapper=None, event_var="_ev0", stream_idx=0)
        output = self._codegen_to_str(line)
        self.assertIn(".wait_event(_ev0)", output)

    def test_sync_marker_line_codegen(self):
        # Use StreamSynchronizeLine as inner to avoid V.graph dependency
        inner = StreamSynchronizeLine(wrapper=None, stream_idx=0)
        marker = SyncMarkerLine(
            wrapper=None,
            inner=inner,
            sync_id=0,
            graph_id=0,
            kind="device_sync",
            src_launch_ids=(3,),
        )
        output = self._codegen_to_str(marker)
        self.assertIn("record_function('ws_sync:0:0:device_sync:3')", output)
        self.assertIn(".synchronize()", output)
        self.assertIn("try:", output)
        self.assertIn("finally:", output)

    def test_sync_marker_line_multiple_launch_ids(self):
        inner = StreamSynchronizeLine(wrapper=None, stream_idx=0)
        marker = SyncMarkerLine(
            wrapper=None,
            inner=inner,
            sync_id=1,
            graph_id=2,
            kind="device_sync",
            src_launch_ids=(3, 5, 7),
        )
        output = self._codegen_to_str(marker)
        self.assertIn("ws_sync:2:1:device_sync:3,5,7", output)

    def test_sync_marker_line_empty_launch_ids(self):
        inner = StreamSynchronizeLine(wrapper=None, stream_idx=0)
        marker = SyncMarkerLine(
            wrapper=None,
            inner=inner,
            sync_id=0,
            graph_id=0,
            kind="device_sync",
            src_launch_ids=(),
        )
        output = self._codegen_to_str(marker)
        self.assertIn("ws_sync:0:0:device_sync:", output)

    def test_sync_marker_frontier_uses_event_record_for_wait_event(self):
        fake_wrapper = type("FakeWrapper", (), {})()
        fake_wrapper.lines = [
            KernelCallLine(
                wrapper=None,
                kernel_name="kernel0",
                call_args=(),
                raw_keys=(),
                raw_args=(),
                arg_types=[],
                triton=True,
                triton_meta={},
                inductor_meta=None,
                device=torch.device("cuda"),
                graph_name="graph",
                original_fxnode_name="kernel0",
                launch_id=3,
            ),
            EventRecordLine(wrapper=None, event_var="_ev0", stream_idx=0),
            WaitEventLine(wrapper=None, event_var="_ev0", stream_idx=0),
        ]
        fake_graph = type("FakeGraph", (), {"graph_id": 7})()

        with V.set_graph_handler(fake_graph):
            PythonWrapperCodegen._wrap_sync_lines_with_markers(fake_wrapper)

        self.assertIsInstance(fake_wrapper.lines[-1], SyncMarkerLine)
        self.assertEqual(fake_wrapper.lines[-1].kind, "wait_event")
        self.assertEqual(fake_wrapper.lines[-1].src_launch_ids, (3,))

    def test_sync_marker_frontier_uses_all_prior_launches_for_device_sync(self):
        fake_wrapper = type("FakeWrapper", (), {})()
        fake_wrapper.lines = [
            KernelCallLine(
                wrapper=None,
                kernel_name="kernel0",
                call_args=(),
                raw_keys=(),
                raw_args=(),
                arg_types=[],
                triton=True,
                triton_meta={},
                inductor_meta=None,
                device=torch.device("cuda"),
                graph_name="graph",
                original_fxnode_name="kernel0",
                launch_id=3,
            ),
            KernelCallLine(
                wrapper=None,
                kernel_name="kernel1",
                call_args=(),
                raw_keys=(),
                raw_args=(),
                arg_types=[],
                triton=True,
                triton_meta={},
                inductor_meta=None,
                device=torch.device("cuda"),
                graph_name="graph",
                original_fxnode_name="kernel1",
                launch_id=5,
            ),
            DeviceSynchronizeLine(wrapper=None),
        ]
        fake_graph = type("FakeGraph", (), {"graph_id": 7})()

        with V.set_graph_handler(fake_graph):
            PythonWrapperCodegen._wrap_sync_lines_with_markers(fake_wrapper)

        self.assertIsInstance(fake_wrapper.lines[-1], SyncMarkerLine)
        self.assertEqual(fake_wrapper.lines[-1].kind, "device_sync")
        self.assertEqual(fake_wrapper.lines[-1].src_launch_ids, (3, 5))

    def test_sync_marker_frontier_uses_all_prior_launches_for_stream_sync(self):
        fake_wrapper = type("FakeWrapper", (), {})()
        fake_wrapper.lines = [
            KernelCallLine(
                wrapper=None,
                kernel_name="kernel0",
                call_args=(),
                raw_keys=(),
                raw_args=(),
                arg_types=[],
                triton=True,
                triton_meta={},
                inductor_meta=None,
                device=torch.device("cuda"),
                graph_name="graph",
                original_fxnode_name="kernel0",
                launch_id=3,
            ),
            KernelCallLine(
                wrapper=None,
                kernel_name="kernel1",
                call_args=(),
                raw_keys=(),
                raw_args=(),
                arg_types=[],
                triton=True,
                triton_meta={},
                inductor_meta=None,
                device=torch.device("cuda"),
                graph_name="graph",
                original_fxnode_name="kernel1",
                launch_id=5,
            ),
            StreamSynchronizeLine(wrapper=None, stream_idx=0),
        ]
        fake_graph = type("FakeGraph", (), {"graph_id": 7})()

        with V.set_graph_handler(fake_graph):
            PythonWrapperCodegen._wrap_sync_lines_with_markers(fake_wrapper)

        self.assertIsInstance(fake_wrapper.lines[-1], SyncMarkerLine)
        self.assertEqual(fake_wrapper.lines[-1].kind, "stream_sync")
        self.assertEqual(fake_wrapper.lines[-1].src_launch_ids, (3, 5))

    def test_sync_marker_frontier_uses_event_record_for_event_sync(self):
        fake_wrapper = type("FakeWrapper", (), {})()
        fake_wrapper.lines = [
            KernelCallLine(
                wrapper=None,
                kernel_name="kernel1",
                call_args=(),
                raw_keys=(),
                raw_args=(),
                arg_types=[],
                triton=True,
                triton_meta={},
                inductor_meta=None,
                device=torch.device("cuda"),
                graph_name="graph",
                original_fxnode_name="kernel1",
                launch_id=1,
            ),
            KernelCallLine(
                wrapper=None,
                kernel_name="kernel2",
                call_args=(),
                raw_keys=(),
                raw_args=(),
                arg_types=[],
                triton=True,
                triton_meta={},
                inductor_meta=None,
                device=torch.device("cuda"),
                graph_name="graph",
                original_fxnode_name="kernel2",
                launch_id=2,
            ),
            EventRecordLine(wrapper=None, event_var="_ev0", stream_idx=0),
            KernelCallLine(
                wrapper=None,
                kernel_name="kernel3",
                call_args=(),
                raw_keys=(),
                raw_args=(),
                arg_types=[],
                triton=True,
                triton_meta={},
                inductor_meta=None,
                device=torch.device("cuda"),
                graph_name="graph",
                original_fxnode_name="kernel3",
                launch_id=3,
            ),
            EventSynchronizeLine(wrapper=None, event_var="_ev0"),
        ]
        fake_graph = type("FakeGraph", (), {"graph_id": 7})()

        with V.set_graph_handler(fake_graph):
            PythonWrapperCodegen._wrap_sync_lines_with_markers(fake_wrapper)

        self.assertIsInstance(fake_wrapper.lines[-1], SyncMarkerLine)
        self.assertEqual(fake_wrapper.lines[-1].kind, "event_sync")
        self.assertEqual(fake_wrapper.lines[-1].src_launch_ids, (1, 2))


if HAS_GPU:

    class TestWeightStreamRuntime(TestCase):
        def setUp(self):
            super().setUp()
            self.schedule = IOSchedule(
                tensors={
                    "w": TensorEntry(
                        name="w",
                        kind="WEIGHT",
                        size_bytes=256,
                        dtype="float32",
                        shape=[8, 8],
                    )
                },
            )
            self.device = torch.device("cuda")

        def tearDown(self):
            WeightStreamRuntime.reset()
            super().tearDown()

        def test_initialize_and_instance(self):
            rt = WeightStreamRuntime.initialize(self.schedule, self.device)
            self.assertIs(WeightStreamRuntime.instance(), rt)
            self.assertEqual(rt._location["w"], "ssd")

        def test_reset(self):
            WeightStreamRuntime.initialize(self.schedule, self.device)
            WeightStreamRuntime.reset()
            with self.assertRaises(AssertionError):
                WeightStreamRuntime.instance()

        def test_evict_vram_frees_gpu_memory(self):
            rt = WeightStreamRuntime.initialize(self.schedule, self.device)
            rt._location["w"] = "vram"
            gpu_tensor = torch.randn(8, 8, device="cuda")
            rt._vram_tensors["w"] = gpu_tensor

            external_ref = gpu_tensor
            rt.evict_vram("w")
            self.assertEqual(rt._location["w"], "dram")
            self.assertNotIn("w", rt._vram_tensors)
            self.assertEqual(external_ref.untyped_storage().size(), 0)

        def test_evict_dram_to_ssd(self):
            rt = WeightStreamRuntime.initialize(self.schedule, self.device)
            rt._location["w"] = "dram"
            rt._dram_buffers["w"] = torch.randn(8, 8).pin_memory()
            rt.evict_dram("w")
            self.assertEqual(rt._location["w"], "ssd")
            self.assertNotIn("w", rt._dram_buffers)

        def test_evict_vram_then_dram_separately(self):
            """VRAM and DRAM evictions are independent operations."""
            rt = WeightStreamRuntime.initialize(self.schedule, self.device)
            rt._location["w"] = "vram"
            rt._vram_tensors["w"] = torch.randn(8, 8, device="cuda")
            rt._dram_buffers["w"] = torch.randn(8, 8).pin_memory()

            # After VRAM evict: GPU freed, but DRAM still present
            rt.evict_vram("w")
            self.assertEqual(rt._location["w"], "dram")
            self.assertIn("w", rt._dram_buffers)

            # After DRAM evict: fully evicted
            rt.evict_dram("w")
            self.assertEqual(rt._location["w"], "ssd")
            self.assertNotIn("w", rt._dram_buffers)

        def test_h2d_prefetch_and_wait(self):
            rt = WeightStreamRuntime.initialize(self.schedule, self.device)
            cpu_tensor = torch.randn(8, 8).pin_memory()
            rt._dram_buffers["w"] = cpu_tensor
            rt._location["w"] = "dram"

            gpu_tensor = rt.h2d_prefetch("w")
            self.assertEqual(gpu_tensor.device.type, "cuda")
            self.assertEqual(rt._location["w"], "vram")

            result = rt.wait_h2d("w")
            self.assertEqual(result.device.type, "cuda")
            torch.testing.assert_close(result.cpu(), cpu_tensor)
            self.assertNotIn("w", rt._h2d_events)

        def test_h2d_uses_separate_stream(self):
            rt = WeightStreamRuntime.initialize(self.schedule, self.device)
            self.assertNotEqual(
                rt._h2d_stream, torch.cuda.current_stream(self.device)
            )

        def test_register_dram_tensor(self):
            rt = WeightStreamRuntime.initialize(self.schedule, self.device)
            cpu_tensor = torch.randn(8, 8)
            rt.register_dram_tensor("w", cpu_tensor)
            self.assertEqual(rt._location["w"], "dram")
            self.assertIs(rt._dram_buffers["w"], cpu_tensor)

        def test_cold_start_prefetch_with_dram(self):
            rt = WeightStreamRuntime.initialize(self.schedule, self.device)
            cpu_tensor = torch.randn(8, 8)
            rt.register_dram_tensor("w", cpu_tensor)
            rt.cold_start_prefetch("w")
            self.assertEqual(rt._location["w"], "dram")

        def test_full_ssd_to_vram_pipeline(self):
            """Test SSD->DRAM->VRAM with real file I/O."""
            weight_data = torch.randn(8, 8)
            with tempfile.NamedTemporaryFile(delete=False, suffix=".bin") as f:
                fpath = f.name
                f.write(weight_data.numpy().tobytes())

            try:
                entry = TensorEntry(
                    name="w",
                    kind="WEIGHT",
                    file=fpath,
                    file_offset=0,
                    size_bytes=8 * 8 * 4,
                    dtype="float32",
                    shape=[8, 8],
                )
                schedule = IOSchedule(tensors={"w": entry})
                rt = WeightStreamRuntime.initialize(schedule, self.device)

                rt.ssd_prefetch("w")
                self.assertIn("w", rt._inflight_reads)

                gpu = rt.h2d_prefetch("w")
                self.assertNotIn("w", rt._inflight_reads)

                result = rt.wait_h2d("w")
                torch.testing.assert_close(
                    result.cpu(), weight_data, atol=0, rtol=0
                )
            finally:
                os.unlink(fpath)


    class TestMarkerInjectionOrdering(TestCase):
        """Markers must not hide compute lines from the injection pass."""

        def test_markers_wrap_after_injection(self):
            """Verify _generate() applies injection before markers.

            In _generate(), the order is:
            1. _inject_weight_streaming_io() (sees raw compute lines)
            2. LaunchMarkerLine wrapping (wraps compute lines after)

            We test by enabling markers only (no schedule) and checking
            that the generated code contains marker + kernel call inside
            the try block.
            """
            import torch._inductor.config as inductor_config

            torch._dynamo.reset()
            model = torch.nn.Linear(10, 10).cuda()
            x = torch.randn(2, 10).cuda()

            out_dir = tempfile.mkdtemp(prefix="ws_marker_test_")
            try:
                inductor_config.weight_streaming_emit_launch_markers = True
                inductor_config.weight_streaming_emit_ids = True
                inductor_config.weight_streaming_output_code = out_dir

                compiled = torch.compile(model)
                out = compiled(x)
                self.assertEqual(out.shape, torch.Size([2, 10]))

                # Find the launch map to confirm launch IDs were assigned
                lmap_files = [
                    f for f in os.listdir(out_dir)
                    if f.startswith("compiled_launch_map_graph")
                ]
                self.assertTrue(len(lmap_files) > 0)
                with open(os.path.join(out_dir, lmap_files[0])) as f:
                    lmap = json.load(f)
                self.assertGreater(len(lmap["launches"]), 0)
            finally:
                inductor_config.weight_streaming_emit_launch_markers = False
                inductor_config.weight_streaming_emit_ids = False
                inductor_config.weight_streaming_output_code = ""

    class TestSidecarEmission(TestCase):
        """Sidecar files written when emit_ids is True."""

        def test_sidecars_written(self):
            import torch._inductor.config as inductor_config

            torch._dynamo.reset()
            model = torch.nn.Linear(10, 10).cuda()
            x = torch.randn(2, 10).cuda()

            out_dir = tempfile.mkdtemp(prefix="ws_sidecar_test_")
            try:
                inductor_config.weight_streaming_emit_ids = True
                inductor_config.weight_streaming_output_code = out_dir

                compiled = torch.compile(model)
                compiled(x)

                files = sorted(os.listdir(out_dir))
                self.assertIn(
                    "compiled_weight_streaming_index.json", files
                )

                # Find launch and tensor map files (graph_id varies)
                lmap_files = [
                    f for f in files
                    if f.startswith("compiled_launch_map_graph")
                ]
                tmap_files = [
                    f for f in files
                    if f.startswith("compiled_tensor_map_graph")
                ]
                self.assertEqual(len(lmap_files), 1)
                self.assertEqual(len(tmap_files), 1)

                # Launch map has entries with launch_id
                with open(
                    os.path.join(out_dir, lmap_files[0])
                ) as f:
                    lmap = json.load(f)
                self.assertGreater(len(lmap["launches"]), 0)
                self.assertIn(
                    "compiled_launch_id", lmap["launches"][0]
                )
                self.assertTrue(len(lmap["compilation_hash"]) > 0)

                # Tensor map has weight entries
                with open(
                    os.path.join(out_dir, tmap_files[0])
                ) as f:
                    tmap = json.load(f)
                self.assertGreater(len(tmap["tensors"]), 0)
                self.assertEqual(
                    tmap["compilation_hash"], lmap["compilation_hash"]
                )

                # Tensor entries have used_by_launch_ids
                for entry in tmap["tensors"]:
                    self.assertIn("used_by_launch_ids", entry)
                    self.assertIsInstance(
                        entry["used_by_launch_ids"], list
                    )
                    # The weight is used by at least one kernel
                    self.assertGreater(
                        len(entry["used_by_launch_ids"]), 0
                    )

                # Index has one graph entry
                with open(
                    os.path.join(
                        out_dir,
                        "compiled_weight_streaming_index.json",
                    )
                ) as f:
                    idx = json.load(f)
                self.assertEqual(len(idx["graphs"]), 1)
            finally:
                inductor_config.weight_streaming_emit_ids = False
                inductor_config.weight_streaming_output_code = ""

    class TestLegacyVramInjection(TestCase):
        """Legacy VRAM ops without launch IDs should still be emitted."""

        def test_legacy_vram_ops_are_not_dropped(self):
            import torch._inductor.config as inductor_config

            torch._dynamo.reset()
            x = torch.randn(2, 10, device="cuda")

            out_dir = tempfile.mkdtemp(prefix="ws_legacy_vram_test_")
            schedule_data = {
                "nodes": [
                    {
                        "idx": 0,
                        "name": "triton_fused_mm_0",
                        "resource_kind": "gpu_stream",
                    }
                ],
                "io_operations": [
                    {
                        "type": "vram_prefetch_h2d",
                        "tensor_name": "w",
                        "before_node": 0,
                    },
                    {
                        "type": "vram_evict_d2h",
                        "tensor_name": "w",
                        "after_node": 0,
                    },
                ],
                "spill_decisions": [],
                "cold_start_prefetches": [],
                "tensor_metadata": {
                    "w": {
                        "kind": "WEIGHT",
                        "size_bytes": 400,
                        "dtype": "float32",
                        "shape": [10, 10],
                    }
                },
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(schedule_data, f)
                f.flush()
                schedule_path = f.name

            try:
                schedule = load_io_schedule(schedule_path)
                WeightStreamRuntime.initialize(schedule, torch.device("cuda"))

                inductor_config.weight_streaming_plan = schedule_path
                inductor_config.weight_streaming_output_code = out_dir

                compiled = torch.compile(lambda t: t + 1)
                out = compiled(x)
                self.assertEqual(out.shape, torch.Size([2, 10]))

                generated_path = os.path.join(
                    out_dir, "weight_streaming_generated.py"
                )
                self.assertTrue(os.path.exists(generated_path))
                with open(generated_path) as f:
                    generated = f.read()

                sync_idx = generated.find("torch.cuda.synchronize()")
                evict_idx = generated.find("_ws_rt.evict_vram('w')")
                self.assertGreaterEqual(sync_idx, 0)
                self.assertGreaterEqual(evict_idx, 0)
                self.assertLess(sync_idx, evict_idx)
                self.assertIn("_ws_rt.h2d_prefetch('w')", generated)
                self.assertIn("_ws_rt.evict_vram('w')", generated)
            finally:
                WeightStreamRuntime.reset()
                inductor_config.weight_streaming_plan = ""
                inductor_config.weight_streaming_output_code = ""
                os.unlink(schedule_path)

        def test_generated_code_contains_launch_and_sync_markers(self):
            import torch._inductor.config as inductor_config

            torch._dynamo.reset()
            x = torch.randn(2, 10, device="cuda")

            out_dir = tempfile.mkdtemp(prefix="ws_sync_marker_test_")
            schedule_data = {
                "nodes": [
                    {
                        "idx": 0,
                        "name": "triton_fused_mm_0",
                        "resource_kind": "gpu_stream",
                    }
                ],
                "io_operations": [
                    {
                        "type": "vram_evict_d2h",
                        "tensor_name": "w",
                        "after_node": 0,
                    },
                ],
                "spill_decisions": [],
                "cold_start_prefetches": [],
                "tensor_metadata": {
                    "w": {
                        "kind": "WEIGHT",
                        "size_bytes": 400,
                        "dtype": "float32",
                        "shape": [10, 10],
                    }
                },
            }

            with tempfile.NamedTemporaryFile(
                mode="w", suffix=".json", delete=False
            ) as f:
                json.dump(schedule_data, f)
                f.flush()
                schedule_path = f.name

            try:
                schedule = load_io_schedule(schedule_path)
                WeightStreamRuntime.initialize(schedule, torch.device("cuda"))

                inductor_config.weight_streaming_plan = schedule_path
                inductor_config.weight_streaming_output_code = out_dir
                inductor_config.weight_streaming_emit_launch_markers = True
                inductor_config.weight_streaming_emit_sync_markers = True
                inductor_config.weight_streaming_emit_ids = True

                compiled = torch.compile(lambda t: t + 1)
                out = compiled(x)
                self.assertEqual(out.shape, torch.Size([2, 10]))

                generated_path = os.path.join(
                    out_dir, "weight_streaming_generated.py"
                )
                self.assertTrue(os.path.exists(generated_path))
                with open(generated_path) as f:
                    generated = f.read()

                self.assertIn("record_function('ws_launch:", generated)
                self.assertIn("record_function('ws_sync:", generated)
                self.assertIn(":device_sync:", generated)
                self.assertIn("torch.cuda.synchronize()", generated)
            finally:
                WeightStreamRuntime.reset()
                inductor_config.weight_streaming_plan = ""
                inductor_config.weight_streaming_output_code = ""
                inductor_config.weight_streaming_emit_launch_markers = False
                inductor_config.weight_streaming_emit_sync_markers = False
                inductor_config.weight_streaming_emit_ids = False
                os.unlink(schedule_path)


if __name__ == "__main__":
    run_tests()
