"""
run_papermill.py: Chay tat ca notebooks theo thu tu bang papermill.
Dam bao tai lap ket qua (reproducible).

Su dung:
    python scripts/run_papermill.py
    python scripts/run_papermill.py --notebook 01
    python scripts/run_papermill.py --all
"""

import os
import sys
import argparse
import time

try:
    import papermill as pm
except ImportError:
    print("papermill chua duoc cai dat. Chay: pip install papermill")
    sys.exit(1)


NOTEBOOKS_ORDER = [
    "01_eda.ipynb",
    "02_preprocess_feature.ipynb",
    "03_mining_clustering.ipynb",
    "04_modeling.ipynb",
    "04b_semi_supervised.ipynb",
    "05_evaluation_report.ipynb",
]

def get_project_root() -> str:
    return os.path.dirname(os.path.dirname(os.path.abspath(__file__)))


def run_notebook(nb_name: str, notebooks_dir: str, output_dir: str,
                 params: dict = None) -> bool:
    input_path  = os.path.join(notebooks_dir, nb_name)
    output_path = os.path.join(output_dir, nb_name.replace(".ipynb", "_executed.ipynb"))

    if not os.path.exists(input_path):
        print(f"[SKIP] Khong tim thay: {input_path}")
        return False

    print(f"\n>> Chay: {nb_name}")
    t0 = time.time()
    try:
        pm.execute_notebook(
            input_path=input_path,
            output_path=output_path,
            parameters=params or {},
            kernel_name="python3",
            progress_bar=True,
            report_mode=False,
        )
        elapsed = time.time() - t0
        print(f"   OK ({elapsed:.1f}s) -> {output_path}")
        return True
    except Exception as e:
        print(f"   FAILED: {e}")
        return False


def main():
    parser = argparse.ArgumentParser(description="Chay notebooks bang papermill")
    parser.add_argument("--notebook", type=str, default=None,
                        help="Prefix notebook can chay, vi du '01' de chay 01_eda.ipynb")
    parser.add_argument("--all", action="store_true", help="Chay tat ca notebooks")
    parser.add_argument("--skip-semi", action="store_true",
                        help="Bo qua notebook 04b_semi_supervised")
    args = parser.parse_args()

    root         = get_project_root()
    nb_dir       = os.path.join(root, "notebooks")
    output_dir   = os.path.join(root, "outputs", "reports", "executed_notebooks")
    os.makedirs(output_dir, exist_ok=True)

    if args.notebook:
        target = [nb for nb in NOTEBOOKS_ORDER if nb.startswith(args.notebook)]
        if not target:
            print(f"Khong tim thay notebook bat dau bang '{args.notebook}'")
            sys.exit(1)
    else:
        target = NOTEBOOKS_ORDER.copy()

    if args.skip_semi and "04b_semi_supervised.ipynb" in target:
        target.remove("04b_semi_supervised.ipynb")
        print("Bo qua: 04b_semi_supervised.ipynb")

    print(f"Chay {len(target)} notebooks:")
    for nb in target:
        print(f"  - {nb}")

    t_total = time.time()
    results = {}
    for nb_name in target:
        ok = run_notebook(nb_name, nb_dir, output_dir)
        results[nb_name] = "OK" if ok else "FAILED"

    elapsed_total = time.time() - t_total
    print(f"\n{'='*50}")
    print(f"KET QUA CHAY NOTEBOOK ({elapsed_total:.1f}s tong cong):")
    for nb, status in results.items():
        icon = "+" if status == "OK" else "x"
        print(f"  [{icon}] {nb}: {status}")
    print(f"{'='*50}")
    print(f"Outputs: {output_dir}")

    if all(v == "OK" for v in results.values()):
        print("\nTat ca notebooks chay thanh cong! Kiem tra outputs/")
        return 0
    else:
        failed = [k for k, v in results.items() if v == "FAILED"]
        print(f"\nCo loi tai: {failed}")
        return 1


if __name__ == "__main__":
    os.chdir(get_project_root())
    sys.exit(main())
