#!/usr/bin/env python3
"""CPU-only build checks for the pinned QSA PR and upstream-fix detection."""

import ast
import importlib.util
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


PROJECT = Path(__file__).resolve().parents[1]
SCRIPT = PROJECT / "docker/patch_vllm_qwen38_qsa_capture.py"
SPEC = importlib.util.spec_from_file_location("qsa_patcher", SCRIPT)
PATCHER = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(PATCHER)

# Reduced pre-PR source, retaining the real runtime patch contexts. No
# torch/vLLM imports or GPU are required to exercise patch application.
SOURCES = {
    PATCHER.QSA: '''class Qwen4ExpQSAMetadataBuilder:
    def build(
        self, common_prefix_len, common_attn_metadata, fast_build=False,
        qsa_state_is_fresh: torch.Tensor | None = None,
        qsa_num_accepted_tokens: torch.Tensor | None = None,
        qsa_is_prefilling: torch.Tensor | None = None,
    ) -> Qwen4ExpQSAMetadata:
        del common_prefix_len, fast_build
        cm = common_attn_metadata
        return Qwen4ExpQSAMetadata(
            num_actual_tokens=cm.num_actual_tokens,
            max_query_len=cm.max_query_len,
            query_start_loc=cm.query_start_loc,
            max_seq_len=cm.max_seq_len,
            seq_lens=cm.seq_lens,
            block_table=cm.block_table_tensor,
            slot_mapping=cm.slot_mapping,
        )
''',
    PATCHER.STATE: '''class Qwen4ExpAttnMetadata:
    qsa_state_slot_ids: torch.Tensor | None = None
    qsa_state_is_fresh: torch.Tensor | None = None
    qsa_num_accepted_tokens: torch.Tensor | None = None
    qsa_is_prefilling: torch.Tensor | None = None

    def get_extra_attn_kwargs(
        self,
        builder,
        num_reqs,
    ):
        kwargs = {}
        kwargs.update(
            qsa_state_is_fresh=self.qsa_state_is_fresh[:num_reqs],
            qsa_num_accepted_tokens=self.qsa_num_accepted_tokens[:num_reqs],
            qsa_is_prefilling=self.qsa_is_prefilling[:num_reqs],
        )
        return kwargs


class Qwen4ExpModelState:
    def prepare_attn(self, input_batch, for_capture=False):
        model_metadata = Qwen4ExpAttnMetadata(
            qsa_state_is_fresh=qsa_state_is_fresh,
            qsa_num_accepted_tokens=qsa_num_accepted_tokens,
            qsa_is_prefilling=qsa_is_prefilling,
        )
        attn_metadata = build_attn_metadata(
            attn_groups=attn_groups,
        )
''',
    PATCHER.CAPTURE: '''def prepare_inputs_to_capture(
    num_reqs, num_tokens, input_buffers, model_state, block_tables,
    full_cudagraph, uniform_decode_graph=False, pcp_manager=None,
):
    input_batch = InputBatch.make_dummy(num_reqs, num_tokens, input_buffers)
    if pcp_manager is not None:
        input_batch = pcp_manager.prepare_inputs_to_capture(input_batch)

    input_batch.uniform_decode_graph = full_cudagraph and uniform_decode_graph

    block_table_provider = pcp_manager or block_tables
    input_block_tables = block_table_provider.get_dummy_block_tables(num_reqs)
    return model_state.prepare_attn(input_batch)
''',
    PATCHER.BATCH: '''class InputBatch:
    # from dummy query lengths: a mixed graph may be captured with uniform rows.
    uniform_decode_graph: bool = False

    @classmethod
    def make_dummy(
        cls,
        *args,
    ):
        return cls()
''',
}


class QSACapturePatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        subprocess.run(["git", "init", "-q", str(self.root)], check=True)
        for name, source in SOURCES.items():
            path = self.root / name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(source)

    def cli(self, repo=PATCHER.B12X_REPO):
        return subprocess.run(
            [sys.executable, str(SCRIPT), str(self.root), "--repo", repo],
            capture_output=True, text=True,
        )

    def upstream_fixed_checkout(self):
        # Model a fresh source checkout that already contains the PR.
        subprocess.run(
            ["git", "apply", str(PATCHER.PATCH)], cwd=self.root, check=True,
        )

    def assert_skip_unchanged(self):
        before = PATCHER.read_sources(self.root)
        result = self.cli()
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("already present", result.stdout)
        self.assertEqual(before, PATCHER.read_sources(self.root))

    def assert_failure_unchanged(self):
        before = PATCHER.read_sources(self.root)
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn("QSA PR #865 failed", result.stderr)
        self.assertEqual(before, PATCHER.read_sources(self.root))

    def test_applies_runtime_pr_and_preserves_live_context_behavior(self):
        result = self.cli(PATCHER.B12X_REPO + ".git")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("Applied QSA", result.stdout)
        sources = PATCHER.read_sources(self.root)
        self.assertTrue(PATCHER.has_fix(sources))
        for name, source in sources.items():
            compile(source, name, "exec")

        # Execute the patched capture path with CPU stand-ins. Both graph
        # modes must flag capture; an ordinary batch must remain unflagged.
        namespace = {}
        exec(sources[PATCHER.BATCH] + sources[PATCHER.CAPTURE], namespace)
        self.assertFalse(namespace["InputBatch"]().cudagraph_capture)
        for full in (False, True):
            batch = namespace["prepare_inputs_to_capture"](
                8, 80, None, SimpleNamespace(prepare_attn=lambda b: b),
                SimpleNamespace(get_dummy_block_tables=lambda n: ()), full,
            )
            self.assertTrue(batch.cudagraph_capture)

        prepare = PATCHER.definition(
            ast.parse(sources[PATCHER.STATE]), "Qwen4ExpModelState", "prepare_attn",
        )
        limit = next(n.value for n in ast.walk(prepare)
                     if isinstance(n, ast.keyword) and n.arg == "qsa_max_seq_len")
        for full, captured, expected in ((False, False, None),
                                         (False, True, 262144), (True, False, 262144)):
            self.assertEqual(eval(compile(ast.Expression(limit), "fixture", "eval"), {
                "self": SimpleNamespace(max_model_len=262144), "for_capture": full,
                "input_batch": SimpleNamespace(cudagraph_capture=captured),
            }), expected)

    def test_skips_pr_already_in_fresh_checkout(self):
        self.upstream_fixed_checkout()
        self.assert_skip_unchanged()

    def test_recognizes_reformatted_upstream_fix(self):
        self.upstream_fixed_checkout()
        for name, source in PATCHER.read_sources(self.root).items():
            (self.root / name).write_text(ast.unparse(ast.parse(source)) + "\n")
        self.assert_skip_unchanged()

    def test_skips_equivalent_eager_boundaries_on_both_entry_paths(self):
        for wrappers in (False, True):
            with self.subTest(wrappers=wrappers):
                source = SOURCES[PATCHER.QSA]
                source += "\nfrom vllm.compilation.breakable_cudagraph import eager_break_during_capture\n"
                if wrappers:
                    source += "\n@eager_break_during_capture\ndef qwen4_exp_b12x_qsa_with_output(): pass\n"
                    source += "\n@eager_break_during_capture\ndef _qsa_run_projected(): pass\n"
                else:
                    source += "\nclass Qwen4ExpQSAAttention:\n"
                    source += "    @eager_break_during_capture\n    def _run_qsa(self): pass\n"
                    source += "    @eager_break_during_capture\n    def _run_projected_qsa(self): pass\n"
                (self.root / PATCHER.QSA).write_text(source)
                self.assert_skip_unchanged()

    def test_one_eager_boundary_does_not_certify_fix(self):
        sources = dict(SOURCES)
        sources[PATCHER.QSA] += (
            "\nfrom vllm.compilation.breakable_cudagraph import eager_break_during_capture\n"
            "class Qwen4ExpQSAAttention:\n"
            "    @eager_break_during_capture\n    def _run_qsa(self): pass\n"
            "    def _run_projected_qsa(self): pass\n"
        )
        self.assertFalse(PATCHER.has_fix(sources))

    def test_upstream_main_is_skipped_without_reading_sources(self):
        for path in self.root.rglob("*.py"):
            path.unlink()
        result = self.cli("https://github.com/vllm-project/vllm.git")
        self.assertEqual(result.returncode, 0, result.stderr)
        self.assertIn("not the B12X", result.stdout)

    def test_conflict_fails_without_partial_application(self):
        path = self.root / PATCHER.CAPTURE
        path.write_text(path.read_text().replace(
            "block_table_provider = pcp_manager or block_tables", "block_table_provider = new_provider()",
        ))
        self.assert_failure_unchanged()

    def test_partial_fix_is_not_repaired(self):
        path = self.root / PATCHER.BATCH
        path.write_text(path.read_text().replace(
            "    @classmethod", "    cudagraph_capture: bool = False\n\n    @classmethod",
        ))
        self.assert_failure_unchanged()

    def test_full_only_capture_marker_is_not_a_complete_fix(self):
        self.upstream_fixed_checkout()
        path = self.root / PATCHER.CAPTURE
        path.write_text(path.read_text().replace(
            "    input_batch.cudagraph_capture = True",
            "    if full_cudagraph:\n        input_batch.cudagraph_capture = True",
        ))
        self.assert_failure_unchanged()

    def test_missing_runtime_file_fails(self):
        (self.root / PATCHER.QSA).unlink()
        result = self.cli()
        self.assertNotEqual(result.returncode, 0)
        self.assertIn(PATCHER.QSA, result.stderr)

    def test_invalid_source_fails(self):
        (self.root / PATCHER.QSA).write_text("def broken(\n")
        self.assert_failure_unchanged()

    def test_postcondition_failure_is_fatal(self):
        with patch.object(PATCHER.subprocess, "run"):
            with self.assertRaisesRegex(ValueError, "postcondition failed"):
                PATCHER.apply_fix(self.root, PATCHER.B12X_REPO)

    def test_dockerfile_applies_patch_only_before_wheel_build(self):
        dockerfile = (PROJECT / "Dockerfile").read_text()
        call = 'RUN python3 /tmp/vllm-patches/patch_vllm_qwen38_qsa_capture.py . --repo "$VLLM_REPO"'
        self.assertEqual(dockerfile.count(call), 1)
        self.assertLess(dockerfile.index("FROM base AS vllm-builder"), dockerfile.index(call))
        self.assertLess(dockerfile.index(call), dockerfile.index("# Prepare build requirements"))
        self.assertIn("COPY docker/vllm-qwen38-qsa-capture-pr865.patch /tmp/vllm-patches/", dockerfile)


if __name__ == "__main__":
    unittest.main()
