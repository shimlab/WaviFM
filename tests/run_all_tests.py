"""
Unified test runner executing both C++ GoogleTests and Python unit/integration tests.
Can be invoked from anywhere via:
    python tests/run_all_tests.py
    python -m tests.run_all_tests
"""

import os
import sys
import re
import time
import inspect
import argparse
import subprocess
import traceback

# Ensure workspace root and build are in sys.path
root_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if root_dir not in sys.path:
    sys.path.insert(0, root_dir)
build_dir = os.path.join(root_dir, "build")
if os.path.exists(build_dir) and build_dir not in sys.path:
    sys.path.insert(0, build_dir)


def find_cpp_test_binary(base_dir):
    """Locate the compiled WaveFactorTests executable across build configurations."""
    candidates = [
        os.path.join(base_dir, "build", "test", "WaveFactorTests.exe"),
        os.path.join(base_dir, "build", "test", "WaveFactorTests"),
        os.path.join(base_dir, "build", "test", "Release", "WaveFactorTests.exe"),
        os.path.join(base_dir, "build", "test", "Debug", "WaveFactorTests.exe"),
        os.path.join(base_dir, "build", "bin", "WaveFactorTests.exe"),
        os.path.join(base_dir, "build", "bin", "WaveFactorTests"),
    ]
    for path in candidates:
        if os.path.isfile(path):
            return path
    return None


def build_cpp_targets(base_dir, verbose=False):
    """Ensure C++ test executable and Python extension are incrementally up-to-date."""
    build_dir = os.path.join(base_dir, "build")
    cmake_cache = os.path.join(build_dir, "CMakeCache.txt")
    if not os.path.isfile(cmake_cache):
        print("Configuring CMake build directory...")
        cmd = ["cmake", "-B", "build", "-S", "."]
        if sys.platform == "win32":
            cmd.extend(["-G", "MinGW Makefiles"])
        res = subprocess.run(cmd, cwd=base_dir, capture_output=not verbose, text=True)
        if res.returncode != 0:
            print(f"[WARN] CMake configuration failed:\n{res.stderr if not verbose else ''}")
            return False

    print("=" * 60)
    print("      CMake Incremental Build (WaveFactorTests & WaveFactor)    ")
    print("=" * 60)
    t0 = time.time()
    for target in ["WaveFactorTests", "WaveFactor"]:
        res = subprocess.run(
            ["cmake", "--build", "build", "--target", target],
            cwd=base_dir,
            capture_output=not verbose,
            text=True,
        )
        if res.returncode != 0:
            print(f"[FAIL] CMake build failed for target '{target}':")
            print(res.stdout or res.stderr)
            return False
        print(f"  OK   Target '{target}' is up-to-date")

    dt = time.time() - t0
    print(f"Build verified in {dt:.2f}s\n")
    return True


def run_cpp_tests(base_dir, verbose=False):
    """Execute the C++ GoogleTest binary and parse the summary results."""
    print("=" * 60)
    print("         C++ GoogleTest Suite (WaveFactorTests)         ")
    print("=" * 60)

    cpp_bin = find_cpp_test_binary(base_dir)
    if not cpp_bin:
        print("[WARN] C++ test binary not found.")
        print("       To compile C++ tests, run:")
        print("         cmake -B build -S .")
        print("         cmake --build build --target WaveFactorTests\n")
        return {"status": "skipped", "passed": 0, "failed": 0, "total": 0, "time": 0.0}

    rel_bin = os.path.relpath(cpp_bin, base_dir)
    print(f"Executable: {rel_bin}\n")

    t0 = time.time()
    try:
        proc = subprocess.run(
            [cpp_bin],
            cwd=base_dir,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as e:
        print(f"[FAIL] Error launching C++ test executable: {e}")
        return {"status": "error", "passed": 0, "failed": 1, "total": 1, "time": time.time() - t0, "output": str(e)}

    dt = time.time() - t0
    output = proc.stdout

    # Parse GoogleTest output
    passed_m = re.search(r"\[\s*PASSED\s*\]\s*(\d+)\s+tests?", output)
    failed_m = re.search(r"\[\s*FAILED\s*\]\s*(\d+)\s+tests?", output)
    total_m = re.search(r"\[==========\]\s*(\d+)\s+tests?", output)

    passed_count = int(passed_m.group(1)) if passed_m else 0
    failed_count = int(failed_m.group(1)) if failed_m else 0
    total_count = int(total_m.group(1)) if total_m else (passed_count + failed_count)

    if verbose or proc.returncode != 0:
        print(output)
    else:
        # Compact summary display
        suite_matches = re.findall(r"\[----------\]\s*(\d+)\s+tests? from (\w+)\s*\(", output)
        for count, suite in suite_matches:
            print(f"  OK   [{suite}] ({count} tests)")
        print(f"\n  C++ Result: {passed_count}/{total_count} passed in {dt:.3f}s")

    if proc.returncode != 0:
        print(f"\n[FAIL] C++ test suite exited with code {proc.returncode}")
        if not verbose:
            print(output)
        return {"status": "failed", "passed": passed_count, "failed": max(1, failed_count), "total": total_count, "time": dt, "output": output}

    print()
    return {"status": "passed", "passed": passed_count, "failed": 0, "total": total_count, "time": dt}


def run_python_tests(base_dir, test_modules, verbose=False):
    """Execute Python test suites using introspected test_* functions."""
    print("=" * 60)
    print("          Python Test Suite (wavefactor package)        ")
    print("=" * 60)

    total_tests = 0
    passed_tests = 0
    failed_tests = 0
    failures = []

    t0 = time.time()
    for mod_name in test_modules:
        print(f"[{mod_name}]")
        try:
            mod = __import__(mod_name, fromlist=["*"])
        except Exception as e:
            print(f"  FAILED to import {mod_name}: {e}")
            traceback.print_exc()
            failed_tests += 1
            failures.append((mod_name, "import", str(e)))
            continue

        functions = [
            (name, func)
            for name, func in inspect.getmembers(mod, inspect.isfunction)
            if name.startswith("test_")
        ]

        for name, func in functions:
            total_tests += 1
            t_sub = time.time()
            try:
                func()
                dt_sub = time.time() - t_sub
                passed_tests += 1
                print(f"  OK   {name} ({dt_sub:.3f}s)")
            except Exception as e:
                dt_sub = time.time() - t_sub
                failed_tests += 1
                failures.append((mod_name, name, traceback.format_exc()))
                print(f"  FAIL {name} ({dt_sub:.3f}s): {e}")

    dt = time.time() - t0
    print(f"\n  Python Result: {passed_tests}/{total_tests} passed in {dt:.3f}s\n")

    if failures:
        print(f"Failures ({len(failures)}):")
        for mod_name, name, err in failures:
            print(f"--- {mod_name}.{name} ---")
            print(err)
        return {"status": "failed", "passed": passed_tests, "failed": failed_tests, "total": total_tests, "time": dt}

    return {"status": "passed", "passed": passed_tests, "failed": 0, "total": total_tests, "time": dt}


def main():
    parser = argparse.ArgumentParser(description="WaveFactor Unified Test Runner (C++ & Python)")
    parser.add_argument("--cpp-only", action="store_true", help="Run only the C++ GoogleTest suite")
    parser.add_argument("--py-only", action="store_true", help="Run only the Python test suite")
    parser.add_argument("--no-build", action="store_true", help="Skip CMake incremental compilation check")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose test execution output")
    args = parser.parse_args()

    # Step 1: Incremental C++ build check (ensures binaries are never stale)
    if not args.no_build:
        build_ok = build_cpp_targets(root_dir, verbose=args.verbose)
        if not build_ok:
            print("[FAIL] Aborting test execution due to compilation failure.")
            sys.exit(1)

    python_test_modules = [
        "tests.test_priors",
        "tests.test_data",
        "tests.test_model",
        "tests.test_reference_parity",
    ]

    start_time = time.time()
    cpp_res = None
    py_res = None

    if not args.py_only:
        cpp_res = run_cpp_tests(root_dir, verbose=args.verbose)

    if not args.cpp_only:
        py_res = run_python_tests(root_dir, python_test_modules, verbose=args.verbose)

    total_elapsed = time.time() - start_time

    # Unified Summary Banner
    print("=" * 60)
    print("                    UNIFIED TEST SUMMARY                    ")
    print("=" * 60)

    total_passed = 0
    total_run = 0
    has_failure = False

    if cpp_res is not None:
        if cpp_res["status"] == "skipped":
            print(f"  - C++ GoogleTests:    SKIPPED (binary not found)")
        elif cpp_res["status"] == "passed":
            print(f"  - C++ GoogleTests:    {cpp_res['passed']}/{cpp_res['total']} PASSED ({cpp_res['time']:.3f}s)")
            total_passed += cpp_res["passed"]
            total_run += cpp_res["total"]
        else:
            print(f"  - C++ GoogleTests:    {cpp_res['failed']}/{cpp_res['total']} FAILED ({cpp_res['time']:.3f}s)")
            total_passed += cpp_res["passed"]
            total_run += cpp_res["total"]
            has_failure = True

    if py_res is not None:
        if py_res["status"] == "passed":
            print(f"  - Python Tests:       {py_res['passed']}/{py_res['total']} PASSED ({py_res['time']:.3f}s)")
            total_passed += py_res["passed"]
            total_run += py_res["total"]
        else:
            print(f"  - Python Tests:       {py_res['failed']}/{py_res['total']} FAILED ({py_res['time']:.3f}s)")
            total_passed += py_res["passed"]
            total_run += py_res["total"]
            has_failure = True

    print("-" * 60)
    print(f"  TOTAL:                {total_passed}/{total_run} PASSED in {total_elapsed:.2f}s")
    print("=" * 60)

    if has_failure:
        print("\n[FAILED] One or more test suites failed.")
        sys.exit(1)
    else:
        print("\n[PASSED] All tests passed successfully!")
        sys.exit(0)


if __name__ == "__main__":
    main()
