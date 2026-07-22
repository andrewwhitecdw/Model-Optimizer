# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse
import importlib.util
import sys
from pathlib import Path

import pytest

SCRIPT = Path(__file__).parents[1] / "scripts" / "benchmark_via_builtin.py"
SPEC = importlib.util.spec_from_file_location("benchmark_via_builtin", SCRIPT)
assert SPEC is not None and SPEC.loader is not None
benchmark = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = benchmark
SPEC.loader.exec_module(benchmark)


@pytest.mark.parametrize("value", ["1", "1,2,3", "a,2", "0,2", "2,-1"])
def test_nk_pair_rejects_invalid_values(value):
    with pytest.raises(argparse.ArgumentTypeError):
        benchmark._nk_pair(value)


@pytest.mark.parametrize("value", ["0", "-1"])
def test_positive_int_rejects_non_positive_values(value):
    with pytest.raises(argparse.ArgumentTypeError, match="positive integer"):
        benchmark._positive_int(value)


@pytest.mark.parametrize(
    "option",
    [
        "--ms",
        "--dry_run_iters",
        "--num_iters",
        "--moe_hidden_size",
        "--moe_intermediate_size",
        "--moe_num_experts",
        "--moe_top_k",
    ],
)
def test_parser_rejects_zero_for_numeric_options(option, capsys):
    with pytest.raises(SystemExit, match="2"):
        benchmark._parser().parse_args(["--flashinfer_repo", "/unused", option, "0"])
    assert "not a positive integer" in capsys.readouterr().err


def test_gemm_cases_are_data_driven_and_preserve_requested_shapes():
    cases = benchmark._gemm_cases([1], [(65, 129)], [])

    assert {case.key.split("_MxNxK=", 1)[0] for case in cases} == {
        "bf16",
        "nvfp4_cudnn",
        "nvfp4_cutlass",
        "nvfp4_cutedsl",
        "nvfp4_trtllm",
        "fp8_cudnn",
        "fp8_cutlass",
        "fp8_trtllm",
    }
    assert len({case.tag for case in cases}) == len(cases)
    assert {case.key.split("_MxNxK=", 1)[1] for case in cases} == {"1x65x129"}
    physical_shapes = {
        case.key.split("_MxNxK=", 1)[0]: (
            case.argv[case.argv.index("--n") + 1],
            case.argv[case.argv.index("--k") + 1],
        )
        for case in cases
    }
    assert physical_shapes["nvfp4_cudnn"] == ("96", "160")
    assert physical_shapes["nvfp4_cutlass"] == ("96", "160")
    assert physical_shapes["nvfp4_cutedsl"] == ("96", "160")
    assert physical_shapes["nvfp4_trtllm"] == ("65", "129")
    assert all(
        shape == ("65", "129")
        for name, shape in physical_shapes.items()
        if not name.startswith("nvfp4_")
    )
    assert {case.quant for case in cases if case.quant is not None} == {
        ("nvfp4_128x4", 1, 160),
        ("nvfp4_8x4", 1, 129),
        ("fp8_static", 1, 129),
    }


def test_nk_names_are_parallel_and_map_duplicate_shapes():
    nks, names_by_nk = benchmark._named_nks(
        [(32, 64), (32, 64), (128, 256)],
        ["o_proj", "out_proj", "qkv_proj"],
    )

    assert nks == [(32, 64), (128, 256)]
    assert names_by_nk == {(32, 64): ["o_proj", "out_proj"], (128, 256): ["qkv_proj"]}
    with pytest.raises(ValueError, match="exactly one name"):
        benchmark._named_nks([(32, 64)], ["o_proj", "out_proj"])


def test_case_tags_are_unique_join_keys():
    case = benchmark._Case("gemm", "duplicate", "bf16_MxNxK=1x2x3", [])

    assert benchmark._index_cases([case]) == {case.tag: case}
    with pytest.raises(RuntimeError, match="duplicate benchmark case tag: duplicate"):
        benchmark._index_cases([case, case])


def test_moe_cases_and_result_combination():
    cases = benchmark._moe_cases([1], (32, 50, 4, 2), [], None)
    assert [case.key.split("_M=", 1)[0] for case in cases] == [
        "bf16_cutlass_moe",
        "fp8_cutlass_moe",
        "nvfp4_cutlass_moe",
        "nvfp4_cutlass_moe_with_quant",
        "fp8_trtllm_moe",
        "nvfp4_trtllm_moe",
        "nvfp4_cutedsl_moe",
    ]
    intermediate_sizes = {
        case.key.split("_M=", 1)[0]: case.argv[case.argv.index("--intermediate_size") + 1]
        for case in cases
    }
    assert intermediate_sizes == {
        "bf16_cutlass_moe": "50",
        "nvfp4_cutlass_moe": "50",
        "nvfp4_cutlass_moe_with_quant": "50",
        "fp8_cutlass_moe": "64",
        "fp8_trtllm_moe": "64",
        "nvfp4_trtllm_moe": "64",
        "nvfp4_cutedsl_moe": "50",
    }
    hidden_sizes = {
        case.key.split("_M=", 1)[0]: case.argv[case.argv.index("--hidden_size") + 1]
        for case in cases
    }
    # Only the trtllm-gen NVFP4 MoE pads hidden (vLLM pads it to 256).
    assert hidden_sizes["nvfp4_trtllm_moe"] == "256"
    assert all(value == "32" for row, value in hidden_sizes.items() if row != "nvfp4_trtllm_moe")
    routines = {
        case.key.split("_M=", 1)[0]: case.argv[case.argv.index("--routine") + 1] for case in cases
    }
    assert routines["fp8_trtllm_moe"] == "trtllm_fp8_per_tensor_scale_moe"
    assert routines["nvfp4_trtllm_moe"] == "trtllm_fp4_block_scale_moe"
    assert routines["nvfp4_cutedsl_moe"] == "cute_dsl_fp4_block_scale_moe"
    # Only the trtllm-gen rows route in-kernel; they use a fixed renormalize
    # method so rows stay comparable across models.
    for case in cases:
        row = case.key.split("_M=", 1)[0]
        if row.endswith("_trtllm_moe"):
            assert case.argv[case.argv.index("--routing_method") + 1] == "renormalize"
        else:
            assert "--routing_method" not in case.argv
    quants = {case.key.split("_M=", 1)[0]: case.quant for case in cases}
    # FP8 rows share the static activation-quant timing; the trtllm-gen and
    # CuteDSL NVFP4 rows use the linear-scale-layout quantize their kernels
    # consume (trtllm at the 256-padded hidden). The NVFP4 CUTLASS pair is a
    # direct fused measurement instead.
    assert quants["fp8_cutlass_moe"] == ("fp8_static", 1, 32)
    assert quants["fp8_trtllm_moe"] == ("fp8_static", 1, 32)
    assert quants["nvfp4_trtllm_moe"] == ("nvfp4_linear", 1, 256)
    assert quants["nvfp4_cutedsl_moe"] == ("nvfp4_linear", 1, 32)
    assert quants["bf16_cutlass_moe"] is None
    assert quants["nvfp4_cutlass_moe"] is None
    assert quants["nvfp4_cutlass_moe_with_quant"] is None

    fp8 = next(case for case in cases if case.key == "fp8_cutlass_moe_M=1")
    assert fp8.quant is not None
    rows = [{"case_tag": fp8.tag, "median_time": "0.001"}]
    quant = {fp8.quant: 2.0}

    results = benchmark._combine({fp8.tag: fp8}, rows, quant)

    assert results["moe"]["fp8_cutlass_moe_M=1"] == 1.0
    assert results["moe"]["fp8_cutlass_moe_with_quant_M=1"] == 3.0


def test_swiglustep_uses_gated_fp8_alignment():
    cases = benchmark._moe_cases([1], (32, 50, 4, 2), [], "SwigluStep")

    fp8 = next(case for case in cases if case.key == "fp8_cutlass_moe_M=1")
    assert fp8.argv[fp8.argv.index("--intermediate_size") + 1] == "64"


def test_non_gated_moe_pads_fp8_and_nvfp4_intermediate_to_128():
    cases = benchmark._moe_cases([1], (32, 50, 4, 2), [], "Relu2")

    intermediate_sizes = {
        case.key.split("_M=", 1)[0]: case.argv[case.argv.index("--intermediate_size") + 1]
        for case in cases
    }
    # The Swiglu-only CuteDSL MoE row is not emitted for non-gated activations.
    assert intermediate_sizes == {
        "bf16_cutlass_moe": "50",
        "fp8_cutlass_moe": "128",
        "nvfp4_cutlass_moe": "128",
        "nvfp4_cutlass_moe_with_quant": "128",
        "fp8_trtllm_moe": "128",
        "nvfp4_trtllm_moe": "128",
    }


def test_unavailable_fp8_quantization_is_written_as_an_error(monkeypatch, capsys, tmp_path):
    case = benchmark._Case(
        section="gemm",
        tag="gemm_fp8_cutlass_MxNxK=1x32x64",
        key="fp8_cutlass_MxNxK=1x32x64",
        argv=[],
        quant=("fp8_static", 1, 64),
    )
    monkeypatch.setattr(benchmark, "vllm_ops", None)

    quant = benchmark._quant_times([case], 1, 1, False)
    results = benchmark._combine(
        {case.tag: case}, [{"case_tag": case.tag, "median_time": "0.001"}], quant
    )
    output = tmp_path / "combined_results.csv"
    benchmark._write_results(output, results, [1], [(32, 64)])

    assert quant == {case.quant: benchmark._FP8_QUANT_UNAVAILABLE}
    assert "[WARN] vLLM is unavailable for FP8 activation quantization" in capsys.readouterr().out
    assert results["gemm"][case.key] == 1.0
    assert (
        results["gemm"]["fp8_cutlass_with_quant_MxNxK=1x32x64"] == benchmark._FP8_QUANT_UNAVAILABLE
    )
    assert f"32x64,1,32,64,fp8_cutlass,True,{benchmark._FP8_QUANT_UNAVAILABLE}\n" in (
        output.read_text()
    )


def test_driver_errors_are_added_to_kernel_and_with_quant_rows(tmp_path):
    case = benchmark._Case(
        section="gemm",
        tag="gemm_fp8_trtllm_MxNxK=8x1280x2880",
        key="fp8_trtllm_MxNxK=8x1280x2880",
        argv=[],
        quant=("fp8_static", 8, 2880),
    )
    output = [
        f"[ERROR] Error running test: --routine mm_fp8 --case_tag {case.tag}\n",
        "[ERROR] Error: K must be divisible by 128, got 2880\n",
    ]

    errors = benchmark._parse_driver_errors(output)
    results = benchmark._combine({case.tag: case}, [], {}, errors)
    csv_path = tmp_path / "combined_results.csv"
    benchmark._write_results(csv_path, results, [8], [(1280, 2880)])

    expected = "ERROR: K must be divisible by 128; got 2880"
    assert errors == {case.tag: "K must be divisible by 128; got 2880"}
    assert results["gemm"][case.key] == expected
    assert results["gemm"]["fp8_trtllm_with_quant_MxNxK=8x1280x2880"] == expected
    assert f"1280x2880,8,1280,2880,fp8_trtllm,False,{expected}\n" in csv_path.read_text()
    assert f"1280x2880,8,1280,2880,fp8_trtllm,True,{expected}\n" in csv_path.read_text()


def test_empty_driver_error_has_no_synthetic_reason():
    tag = "gemm_nvfp4_cutlass_MxNxK=8x2880x1024"
    output = [
        f"[ERROR] Error running test: --routine mm_fp4 --case_tag {tag}\n",
        "[ERROR] Error:\n",
    ]

    assert benchmark._parse_driver_errors(output) == {tag: ""}


def test_write_results_emits_long_form_rows(tmp_path):
    output = tmp_path / "combined_results.csv"
    benchmark._write_results(
        output,
        {
            "gemm": {
                "bf16_MxNxK=1x2x3": 1.25,
                "fp8_cutlass_MxNxK=8x4x5": 3.5,
                "fp8_cutlass_with_quant_MxNxK=8x4x5": 4.5,
            },
            "moe": {"fp8_cutlass_moe_with_quant_M=8": 2.5},
        },
        [1, 8],
        [(4, 5), (2, 3)],
        {(4, 5): ["q_proj|k_proj|v_proj"], (2, 3): ["in_proj", "out_proj"]},
        header="flashinfer test-header",
        moe_label="H=32 F=50 E=4 top_k=2 activation=Relu2",
    )

    assert output.read_text() == (
        "flashinfer test-header\n"
        "GEMM\n"
        "module_name,M,N,K,backend,with_quant,runtime\n"
        "q_proj|k_proj|v_proj,8,4,5,fp8_cutlass,False,3.500\n"
        "q_proj|k_proj|v_proj,8,4,5,fp8_cutlass,True,4.500\n"
        "in_proj,1,2,3,bf16,False,1.250\n"
        "out_proj,1,2,3,bf16,False,1.250\n"
        "\n"
        "MoE\n"
        "H=32 F=50 E=4 top_k=2 activation=Relu2\n"
        "module_name,M,N,K,backend,with_quant,runtime\n"
        "experts,8,,,fp8_cutlass,True,2.500\n"
    )


def test_top_k_cannot_exceed_expert_count(monkeypatch, capsys):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--flashinfer_repo",
            "/unused",
            "--moe_hidden_size",
            "4",
            "--moe_intermediate_size",
            "8",
            "--moe_num_experts",
            "1",
            "--moe_top_k",
            "2",
        ],
    )

    with pytest.raises(SystemExit, match="2"):
        benchmark.main()
    assert "--moe_top_k cannot exceed --moe_num_experts" in capsys.readouterr().err


@pytest.mark.parametrize(
    ("returncode", "expected_reason"),
    [
        (0, "FlashInfer produced no result row"),
        (1, "FlashInfer driver exited with status 1"),
    ],
)
def test_missing_builtin_results_still_writes_combined_errors(
    monkeypatch, tmp_path, returncode, expected_reason
):
    benchmarks_dir = tmp_path / "flashinfer" / "benchmarks"
    benchmarks_dir.mkdir(parents=True)
    (benchmarks_dir / "flashinfer_benchmark.py").write_text("")
    workdir = tmp_path / "results"
    monkeypatch.setattr(benchmark, "_run_case", lambda *_: (returncode, []))
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--flashinfer_repo",
            str(benchmarks_dir.parent),
            "--ms",
            "1",
            "--nks",
            "2,3",
            "--workdir",
            str(workdir),
        ],
    )

    with pytest.raises(RuntimeError, match="FlashInfer failed benchmark cases"):
        benchmark.main()

    assert not (workdir / "builtin_results.csv").exists()
    combined = (workdir / "combined_results.csv").read_text()
    assert f"2x3,1,2,3,bf16,False,ERROR: {expected_reason}" in combined
    assert "driver.log" in combined
    # The reproducibility header leads both the combined CSV and driver.log.
    assert combined.splitlines()[0].startswith("flashinfer ")
    assert (workdir / "driver.log").read_text().startswith("flashinfer ")


def test_case_rows_with_foreign_tags_are_treated_as_failures(monkeypatch, tmp_path):
    benchmarks_dir = tmp_path / "flashinfer" / "benchmarks"
    benchmarks_dir.mkdir(parents=True)
    (benchmarks_dir / "flashinfer_benchmark.py").write_text("")
    workdir = tmp_path / "results"

    def fake_run_case(benchmarks_dir, argv, log):
        output = Path(argv[argv.index("--output_path") + 1])
        output.write_text("case_tag,median_time\nsomeone_else,0.001\n")
        return 0, []

    monkeypatch.setattr(benchmark, "_run_case", fake_run_case)
    monkeypatch.setattr(
        sys,
        "argv",
        [
            str(SCRIPT),
            "--flashinfer_repo",
            str(benchmarks_dir.parent),
            "--ms",
            "1",
            "--nks",
            "2,3",
            "--workdir",
            str(workdir),
        ],
    )

    with pytest.raises(RuntimeError, match="FlashInfer failed benchmark cases"):
        benchmark.main()

    combined = (workdir / "combined_results.csv").read_text()
    assert "no result row" in combined


def test_run_case_streams_and_appends_to_the_driver_log(tmp_path, capsys):
    benchmarks_dir = tmp_path / "benchmarks"
    benchmarks_dir.mkdir()
    (benchmarks_dir / "flashinfer_benchmark.py").write_text(
        "print('line one')\nprint('line two')\n"
    )
    driver_log = tmp_path / "driver.log"

    with driver_log.open("w") as log:
        returncode, lines = benchmark._run_case(benchmarks_dir, ["--unused"], log)

    assert returncode == 0
    assert lines == ["line one\n", "line two\n"]
    assert driver_log.read_text() == "line one\nline two\n"
    assert "line one" in capsys.readouterr().out


def test_write_builtin_merges_heterogeneous_row_columns(tmp_path):
    path = tmp_path / "builtin_results.csv"

    benchmark._write_builtin(
        path,
        [
            {"routine": "mm_bf16", "median_time": "0.004", "case_tag": "a"},
            {"routine": "cutlass_fused_moe", "case_tag": "b", "num_experts": "8"},
        ],
    )

    assert path.read_text() == (
        "routine,median_time,case_tag,num_experts\nmm_bf16,0.004,a,\ncutlass_fused_moe,,b,8\n"
    )
