#!/usr/bin/env python3
"""Apply local-inference-lab/vllm PR #865 before building the B12X wheel.

The adjacent patch contains only the four runtime files from PR head
26b42cf9eb715b118e55f91f53fa0ab412a79c09. Unknown or partially fixed source
must fail the build so the upstream change can be reviewed.
"""

import argparse
import ast
import subprocess
from pathlib import Path


B12X_REPO = "https://github.com/local-inference-lab/vllm"
PATCH = Path(__file__).with_name("vllm-qwen38-qsa-capture-pr865.patch")
QSA = "vllm/models/qwen4_exp/nvidia/b12x_qsa.py"
STATE = "vllm/models/qwen4_exp/nvidia/model_state.py"
CAPTURE = "vllm/v1/worker/gpu/cudagraph_utils.py"
BATCH = "vllm/v1/worker/gpu/input_batch.py"


def definition(tree, *names):
    """Find a definition by its lexical path, without matching comments."""
    for name in names:
        matches = [
            node for node in tree.body
            if isinstance(node, (ast.ClassDef, ast.FunctionDef)) and node.name == name
        ]
        if len(matches) != 1:
            raise ValueError(f"expected one definition of {'.'.join(names)}")
        tree = matches[0]
    return tree


def contains(node, code, *, top_level=False):
    expected = ast.dump(ast.parse(code).body[0])
    candidates = node.body if top_level else ast.walk(node)
    return any(ast.dump(candidate) == expected for candidate in candidates)


def keyword(node, name, expression):
    expected = ast.dump(ast.parse(expression, mode="eval").body)
    return any(
        isinstance(item, ast.keyword) and item.arg == name
        and ast.dump(item.value) == expected
        for item in ast.walk(node)
    )


def has_fix(sources):
    """Recognize the PR's complete metadata path or QSA eager boundaries.

    AST comparisons accept rebases and formatting changes. Deliberately do
    not guess about unfamiliar rewrites: a failed patch then requires review.
    """
    trees = {path: ast.parse(source, filename=path) for path, source in sources.items()}
    qsa = trees[QSA]
    imported_break = any(
        isinstance(node, ast.ImportFrom)
        and node.module == "vllm.compilation.breakable_cudagraph"
        and any(alias.name == "eager_break_during_capture" and alias.asname is None
                for alias in node.names)
        for node in qsa.body
    )
    if imported_break:
        # Cover normal QSA and the projected/reused MTP read path. One
        # decorated entry point alone is not enough to certify this fix.
        for paths in (
            (("Qwen4ExpQSAAttention", "_run_qsa"),
             ("Qwen4ExpQSAAttention", "_run_projected_qsa")),
            (("qwen4_exp_b12x_qsa_with_output",), ("_qsa_run_projected",)),
        ):
            try:
                if all(any(isinstance(d, ast.Name) and d.id == "eager_break_during_capture"
                           for d in definition(qsa, *path).decorator_list)
                       for path in paths):
                    return True
            except ValueError:
                pass

    try:
        builder = definition(qsa, "Qwen4ExpQSAMetadataBuilder", "build")
        metadata = definition(trees[STATE], "Qwen4ExpAttnMetadata")
        extra = definition(metadata, "get_extra_attn_kwargs")
        prepare = definition(trees[STATE], "Qwen4ExpModelState", "prepare_attn")
        capture = definition(trees[CAPTURE], "prepare_inputs_to_capture")
        batch = definition(trees[BATCH], "InputBatch")
    except ValueError:
        return False
    return all((
        any(arg.arg == "qsa_max_seq_len" for arg in builder.args.args + builder.args.kwonlyargs),
        keyword(builder, "max_seq_len",
                "cm.max_seq_len if qsa_max_seq_len is None else qsa_max_seq_len"),
        contains(metadata, "qsa_max_seq_len: int | None = None", top_level=True),
        keyword(extra, "qsa_max_seq_len", "self.qsa_max_seq_len"),
        keyword(prepare, "qsa_max_seq_len",
                "self.max_model_len if for_capture or input_batch.cudagraph_capture else None"),
        contains(capture, "input_batch.cudagraph_capture = True", top_level=True),
        contains(batch, "cudagraph_capture: bool = False", top_level=True),
    ))


def read_sources(root):
    return {path: (root / path).read_text() for path in (QSA, STATE, CAPTURE, BATCH)}


def apply_fix(root, repo):
    if repo.rstrip("/").removesuffix(".git") != B12X_REPO:
        return "QSA PR #865: not the B12X vLLM fork; skipping"
    root = root.resolve()
    if has_fix(read_sources(root)):
        return "QSA PR #865 or an equivalent QSA graph fix is already present; skipping"
    # Check all files before modifying any. No partial application, fuzzy
    # fallback, or repair of partially applied patches during fresh builds.
    for args in (("--check",), ()):
        subprocess.run(
            ["git", "apply", *args, str(PATCH)], cwd=root, check=True,
        )
    if not has_fix(read_sources(root)):
        raise ValueError("PR #865 applied but the QSA capture fix postcondition failed")
    return "Applied QSA capture-context fix from local-inference-lab/vllm PR #865"


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("source_root", type=Path)
    parser.add_argument("--repo", required=True)
    args = parser.parse_args()
    try:
        print(apply_fix(args.source_root, args.repo))
    except (OSError, SyntaxError, ValueError, subprocess.CalledProcessError) as exc:
        raise SystemExit(
            f"QSA PR #865 failed for {args.source_root}: {exc}. "
            "Review the B12X source and patch diagnostics before rebuilding."
        ) from exc


if __name__ == "__main__":
    main()
