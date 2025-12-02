#!/usr/bin/env python3
"""
tools/test_submit_job.py

Test script to submit a QASM job to the Orchestrator and print the merged result.

Usage:
    python tools/test_submit_job.py --nodes 3 --max-parts 3
    python tools/test_submit_job.py --qasm my_circuit.qasm --nodes 2

This script assumes:
 - Orchestrator running at http://localhost:8000 (adjust ORCH_URL below)
 - QNodes are registered (or will register within the wait timeout)

It writes the full JSON response to out/last_submit_result.json
"""

import argparse
import json
import os
import sys
import time
from typing import Optional
import requests

ORCH_URL = os.getenv("ORCH_URL", "http://localhost:8000")
OUT_DIR = os.path.join(os.getcwd(), "out")
os.makedirs(OUT_DIR, exist_ok=True)

DEFAULT_QASM = """OPENQASM 2.0;
include \"qelib1.inc\";
qreg q[6];
creg c[6];
h q[0];
cx q[0],q[1];
h q[2];
cx q[2],q[3];
h q[4];
cx q[4],q[5];
measure q -> c;
"""

def wait_for_orchestrator(timeout_s: int = 10) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{ORCH_URL}/nodes", timeout=2)
            if r.status_code == 200:
                return
        except Exception:
            pass
        time.sleep(0.5)
    raise RuntimeError(f"Orchestrator not reachable at {ORCH_URL} after {timeout_s}s")

def wait_for_nodes(min_nodes: int, timeout_s: int = 30) -> None:
    deadline = time.time() + timeout_s
    while time.time() < deadline:
        try:
            r = requests.get(f"{ORCH_URL}/nodes", timeout=2)
            if r.status_code == 200:
                data = r.json()
                nodes = data.get("nodes", [])
                if len(nodes) >= min_nodes:
                    print(f"Found {len(nodes)} nodes (>= {min_nodes}) registered in orchestrator.")
                    for n in nodes:
                        print(f" - {n.get('id')} ({n.get('technology')}) api_url={n.get('api_url')}")
                    return
        except Exception as e:
            pass
        time.sleep(1.0)
    raise RuntimeError(f"Timeout waiting for {min_nodes} nodes to register in orchestrator (checked until {timeout_s}s)")

def load_qasm_from_file(path: str) -> str:
    with open(path, "r") as f:
        return f.read()

def submit_job(qasm: str, max_parts: int) -> dict:
    payload = {"qasm": qasm, "max_parts": max_parts}
    r = requests.post(f"{ORCH_URL}/submit_job", json=payload, timeout=120)
    r.raise_for_status()
    return r.json()

def pretty_print_response(resp: dict) -> None:
    print("\n--- ORCHESTRATOR RESPONSE ---")
    print(json.dumps(resp, indent=2))

    merged = resp.get("merged_result")
    if merged:
        print("\n--- MERGED RESULT (counts summary) ---")
        counts = merged.get("counts", {})
        total_shots = merged.get("total_shots", 0)
        print(f"Total shots (estimated): {total_shots}, unique bitstrings: {len(counts)}")
        # print top-10 bitstrings by count
        top = sorted(counts.items(), key=lambda x: -x[1])[:10]
        for bs, c in top:
            print(f" {bs} : {c}")
        print("\nFidelity estimate:", merged.get("fidelity_estimate"))
    else:
        print("\nNo merged_result present. Inspecting node_results for debugging:")
        node_results = resp.get("node_results", [])
        for nr in node_results:
            print(" - slice_id:", nr.get("slice_id"), "status:", nr.get("status"))
            if nr.get("status") == "ok":
                print("   response keys:", list(nr.get("response", {}).keys()))
            else:
                print("   reason:", nr.get("reason"))

def save_response(resp: dict):
    path = os.path.join(OUT_DIR, "last_submit_result.json")
    with open(path, "w") as f:
        json.dump(resp, f, indent=2)
    print(f"\nSaved full response to {path}")

def main():
    parser = argparse.ArgumentParser(description="Submit a job to the Orchestrator and show merged_result.")
    parser.add_argument("--qasm", type=str, help="Path to QASM file to submit (optional). If omitted, a small default QASM is used.")
    parser.add_argument("--nodes", type=int, default=1, help="Minimum number of nodes expected to be registered on orchestrator before submit.")
    parser.add_argument("--max-parts", type=int, default=2, help="max_parts to send to /submit_job")
    parser.add_argument("--wait", type=int, default=30, help="Seconds to wait for nodes to register")
    args = parser.parse_args()

    try:
        print(f"Checking orchestrator at {ORCH_URL} ...")
        wait_for_orchestrator(timeout_s=5)
    except Exception as e:
        print("Orchestrator not reachable:", e)
        sys.exit(1)

    if args.nodes > 0:
        try:
            wait_for_nodes(args.nodes, timeout_s=args.wait)
        except Exception as e:
            print("Node wait failed:", e)
            # we continue — sometimes you want to submit anyway
            # sys.exit(1)

    if args.qasm:
        if not os.path.exists(args.qasm):
            print("QASM file not found:", args.qasm)
            sys.exit(1)
        qasm = load_qasm_from_file(args.qasm)
    else:
        qasm = DEFAULT_QASM

    print("Submitting job (max_parts=%d) ..." % args.max_parts)
    try:
        resp = submit_job(qasm, args.max_parts)
    except Exception as e:
        print("Submit failed:", e)
        sys.exit(1)

    pretty_print_response(resp)
    save_response(resp)
    print("\nDone.")

if __name__ == "__main__":
    main()
