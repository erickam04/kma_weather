"""파이프라인 오케스트레이터 - 설정(pipeline_config.py) 기반 일괄 실행.

batch 평가 -> 결과 집계 -> batch 그래프 -> target별 neighbor 시각화를
순서대로 실행한다. target/기간/method를 바꾸려면 pipeline_config.py만 수정하면 된다.

실행:
    python -m scripts.run_pipeline                # 전체 (batch는 12개월 로드라 오래 걸림)
    python -m scripts.run_pipeline --only batch   # 한 단계만
    python -m scripts.run_pipeline --skip batch   # batch 제외 (기존 raw 재사용해 분석/시각화만)

단계 이름: batch | analyze | viz_batch | viz_targets
계획: .omc/plans/pipeline-config-refactor.md
"""

import argparse
import time
import traceback

import pipeline_config as cfg


def step_batch():
    from scripts import run_batch_test
    run_batch_test.main()


def step_analyze():
    from analysis import analyze_results
    analyze_results.main()


def step_viz_batch():
    from visualization import visualize_batch
    visualize_batch.main()


def step_viz_targets():
    from scripts import run_visualize_targets
    run_visualize_targets.run()


def step_viz_maps():
    from visualization import visualize_targets_map
    visualize_targets_map.main()


STEPS = [
    ("batch", step_batch, "IDW batch 평가 (run_batch_test)"),
    ("analyze", step_analyze, "결과 집계 (analyze_results)"),
    ("viz_batch", step_viz_batch, "batch 그래프 (visualize_batch)"),
    ("viz_targets", step_viz_targets, "target별 neighbor 시각화 (run_visualize_targets)"),
    ("viz_maps", step_viz_maps, "target 지도·이웃 고도 시각화 (visualize_targets_map)"),
]
STEP_NAMES = [name for name, _, _ in STEPS]


def parse_args():
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--only", choices=STEP_NAMES, nargs="+",
                   help="지정한 단계만 실행")
    p.add_argument("--skip", choices=STEP_NAMES, nargs="+", default=[],
                   help="지정한 단계 제외")
    return p.parse_args()


def main():
    args = parse_args()
    selected = args.only if args.only else STEP_NAMES
    selected = [s for s in selected if s not in args.skip]

    print("=" * 60)
    print("파이프라인 설정 (pipeline_config.py)")
    print("=" * 60)
    print(f"  TARGET_MODE : {cfg.TARGET_MODE}"
          + (f" (지점 {len(cfg.TARGET_LIST)}개)" if cfg.TARGET_MODE == "explicit" else ""))
    print(f"  YEAR        : {cfg.YEAR}")
    print(f"  METHODS     : {cfg.METHODS}")
    print(f"  OUT_BATCH   : {cfg.OUT_BATCH}")
    print(f"  OUT_VIZ     : {cfg.OUT_VIZ}")
    print(f"  실행 단계   : {selected}")

    results = []
    for name, fn, desc in STEPS:
        if name not in selected:
            continue
        print()
        print("#" * 60)
        print(f"# 단계 [{name}] {desc}")
        print("#" * 60)
        t0 = time.time()
        try:
            fn()
            elapsed = time.time() - t0
            results.append((name, "OK", elapsed))
            print(f"\n[{name}] 완료 ({elapsed:.1f}s)")
        except Exception:
            elapsed = time.time() - t0
            results.append((name, "FAIL", elapsed))
            print(f"\n[{name}] 실패 ({elapsed:.1f}s):")
            traceback.print_exc()
            print("이후 단계를 중단합니다 (이전 단계 산출물에 의존).")
            break

    print()
    print("=" * 60)
    print("파이프라인 요약")
    print("=" * 60)
    for name, status, elapsed in results:
        print(f"  {status:4s}  {name:12s} {elapsed:8.1f}s")
    if any(s == "FAIL" for _, s, _ in results):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
