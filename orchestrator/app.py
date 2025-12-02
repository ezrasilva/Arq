"""
orchestrator_with_merge.py

Versão do Orchestrator Priority A com integração do módulo de merge
(orchestrator_merge.merge_all_slices). Este arquivo é plug-and-play com os
QNodes e módulos de merge que já foram criados no canvas.

Funcionalidades principais:
- /register para QNodes
- /submit_job: particiona (HDH ou fallback), faz placement, reserva EPRs (stub),
  envia slices, coleta respostas e faz merge final (quando disponível)
- /nodes para inspeção

Instruções:
- coloque este arquivo no diretório do orchestrator
- certifique-se de que `orchestrator_merge.py` exista no mesmo diretório
- execute com: uvicorn orchestrator_with_merge:app --reload --port 8000
"""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import threading
import uuid
import logging
import time
import os
import re

# Optional imports (hdh may not be installed in the dev env)
try:
    from hdh.importers import from_qasm
    from hdh.partitioning import partition as hdh_partition
    HDH_AVAILABLE = True
except Exception:
    HDH_AVAILABLE = False

# import merge utilities
try:
    from orchestrator_merge import merge_all_slices
    MERGE_AVAILABLE = True
except Exception:
    MERGE_AVAILABLE = False

import requests

app = FastAPI(title="Orchestrator with Merge - Priority A")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

# --- In-memory registry ---
NODES: Dict[str, Dict[str, Any]] = {}
NODES_LOCK = threading.Lock()

# Config (can be moved to env or file)
SEQUENCE_URL = os.getenv("SEQUENCE_URL", "http://sequence:8000")
MAX_PARTS_DEFAULT = int(os.getenv("MAX_PARTS", "4"))

# --- Pydantic models ---
class NodeRegister(BaseModel):
    id: str
    technology: Optional[str] = None
    num_qubits: int
    native_gates: Optional[List[str]] = []
    location: Optional[str] = None
    api_url: Optional[str] = None  # where to call /execute_slice

class SubmitJobReq(BaseModel):
    qasm: str
    max_parts: Optional[int] = None

class SliceAssignment(BaseModel):
    slice_id: str
    assigned_node: Optional[str]
    qasm: str
    logical_qubits: List[int]
    requirements: Dict[str, Any]
    dependencies: List[str]

# --- Utilities ---

def register_node(node: NodeRegister):
    with NODES_LOCK:
        NODES[node.id] = {
            "id": node.id,
            "technology": node.technology,
            "num_qubits": node.num_qubits,
            "native_gates": set(node.native_gates or []),
            "location": node.location,
            "api_url": node.api_url,
            "available_qubits": node.num_qubits,
            "load": 0,
            "last_seen": time.time()
        }
    logger.info(f"Registered node {node.id}")


def list_nodes() -> List[Dict[str, Any]]:
    with NODES_LOCK:
        # return shallow copy
        return [dict(v) for v in NODES.values()]

# --- HDH Connector / fallback ---

def analyze_and_partition_with_hdh(qasm: str, k: int) -> Dict[str, Any]:
    if not HDH_AVAILABLE:
        raise RuntimeError("HDH package not available in the environment")

    logger.info("Using HDH to import QASM and partition")
    graph = from_qasm(qasm)
    try:
        parts = hdh_partition(graph, k=k)
        subgraphs = getattr(parts, 'subgraphs', None) or getattr(parts, 'parts', None) or parts
        slices = []
        for i, sg in enumerate(subgraphs):
            try:
                qasm_slice = sg.to_qasm()
            except Exception:
                qasm_slice = qasm
            slices.append({
                "slice_id": f"s{i}",
                "qasm": qasm_slice,
                "logical_qubits": list(getattr(sg, 'qubits', [])) if hasattr(sg, 'qubits') else [],
                "requirements": {"native_gates": []},
                "dependencies": []
            })
        return {"job_id": str(uuid.uuid4()), "slices": slices, "global_dependencies": []}
    except TypeError:
        parts = hdh_partition(graph)
        subgraphs = getattr(parts, 'subgraphs', None) or getattr(parts, 'parts', None) or parts
        slices = []
        for i, sg in enumerate(subgraphs):
            try:
                qasm_slice = sg.to_qasm()
            except Exception:
                qasm_slice = qasm
            slices.append({
                "slice_id": f"s{i}",
                "qasm": qasm_slice,
                "logical_qubits": list(getattr(sg, 'qubits', [])) if hasattr(sg, 'qubits') else [],
                "requirements": {"native_gates": []},
                "dependencies": []
            })
        return {"job_id": str(uuid.uuid4()), "slices": slices, "global_dependencies": []}


def fallback_partition(qasm: str, max_slices: int = 2) -> Dict[str, Any]:
    qubits = set(re.findall(r"q\[(\d+)\]", qasm))
    qubits = sorted(int(q) for q in qubits)
    if not qubits:
        qubits = [0]
    groups = [[] for _ in range(max_slices)]
    for i, q in enumerate(qubits):
        groups[i % max_slices].append(q)
    slices = []
    for i, grp in enumerate(groups):
        if not grp:
            continue
        slices.append({
            "slice_id": f"s{i}",
            "qasm": qasm,
            "logical_qubits": grp,
            "requirements": {"native_gates": ["h","cx"]},
            "dependencies": []
        })
    for j in range(len(slices)-1):
        slices[j+1]["dependencies"].append(slices[j]["slice_id"])
    return {"job_id": str(uuid.uuid4()), "slices": slices, "global_dependencies": []}

# --- Placement Scheduler ---

def compute_cost(slice_obj: Dict[str, Any], node: Dict[str, Any]) -> float:
    req_gates = set(slice_obj.get("requirements", {}).get("native_gates", []))
    node_gates = set(node.get("native_gates", set()))
    gate_penalty = 0 if req_gates.issubset(node_gates) or len(req_gates)==0 else 1.0
    need_q = len(slice_obj.get("logical_qubits", []))
    if node.get("available_qubits", 0) < need_q:
        return float('inf')
    load_penalty = node.get("load", 0) / max(1, node.get("num_qubits",1))
    free_qubits_bonus = (node.get("available_qubits",0) - need_q) / max(1, node.get("num_qubits",1))
    cost = gate_penalty * 2.0 + load_penalty - free_qubits_bonus
    return cost


def place_slices(slices: List[Dict[str, Any]], nodes: List[Dict[str, Any]]) -> List[SliceAssignment]:
    sorted_slices = sorted(slices, key=lambda s: -len(s.get('logical_qubits', [])))
    assignments: List[SliceAssignment] = []
    nodes_map = {n['id']: dict(n) for n in nodes}
    for s in sorted_slices:
        best_node = None
        best_cost = float('inf')
        for n in nodes_map.values():
            c = compute_cost(s, n)
            if c < best_cost:
                best_cost = c
                best_node = n
        if best_node is None or best_cost == float('inf'):
            assignments.append(SliceAssignment(
                slice_id=s['slice_id'],
                assigned_node=None,
                qasm=s['qasm'],
                logical_qubits=s['logical_qubits'],
                requirements=s.get('requirements', {}),
                dependencies=s.get('dependencies', [])
            ))
        else:
            need_q = len(s.get('logical_qubits', []))
            best_node['available_qubits'] -= need_q
            best_node['load'] += 1
            assignments.append(SliceAssignment(
                slice_id=s['slice_id'],
                assigned_node=best_node['id'],
                qasm=s['qasm'],
                logical_qubits=s['logical_qubits'],
                requirements=s.get('requirements', {}),
                dependencies=s.get('dependencies', [])
            ))
    return assignments

# --- EPR stub ---

def reserve_epr_pairs(pairs: List[Dict[str, Any]]) -> Dict[str, Any]:
    import random
    reservations = {}
    for p in pairs:
        rid = str(uuid.uuid4())
        latency = random.uniform(5, 50)
        success = random.random() < 0.95
        fidelity = random.uniform(0.85, 0.99) if success else 0.0
        reservations[rid] = {"from": p.get('from'), "to": p.get('to'), "count": p.get('count',1), "status":"reserved" if success else "failed", "latency_ms": latency, "fidelity": fidelity}
    logger.info(f"Reserved EPR pairs (stub): {reservations}")
    return reservations

# --- Execution helpers ---

def send_slice_to_node(assignment: SliceAssignment) -> Dict[str, Any]:
    if not assignment.assigned_node:
        return {"status": "error", "reason": "no_node_assigned", "slice_id": assignment.slice_id}
    node_info = NODES.get(assignment.assigned_node)
    if not node_info:
        return {"status": "error", "reason": "node_not_found", "slice_id": assignment.slice_id}
    api_url = node_info.get("api_url")
    if not api_url:
        return {"status": "error", "reason": "node_missing_api_url", "slice_id": assignment.slice_id}
    payload = {
        "slice_id": assignment.slice_id,
        "qasm": assignment.qasm,
        "logical_qubits": assignment.logical_qubits,
        "requirements": assignment.requirements,
        "dependencies": assignment.dependencies
    }
    try:
        r = requests.post(f"{api_url}/execute_slice", json=payload, timeout=30)
        r.raise_for_status()
        return {"status": "ok", "slice_id": assignment.slice_id, "response": r.json()}
    except Exception as e:
        logger.exception("Failed to send slice to node")
        return {"status": "error", "reason": str(e), "slice_id": assignment.slice_id}

# helper to extract total number of qubits from QASM
def extract_total_qubits(qasm: str) -> int:
    m = re.search(r"qreg\s+\w+\s*\[\s*(\d+)\s*\]", qasm)
    if m:
        return int(m.group(1))
    indices = re.findall(r"q\[(\d+)\]", qasm)
    if indices:
        return max(int(i) for i in indices) + 1
    return 0

# --- FastAPI endpoints ---

@app.post("/register")
def api_register(node: NodeRegister):
    register_node(node)
    return {"status": "ok", "node_id": node.id}


@app.post("/submit_job")
def api_submit_job(req: SubmitJobReq):
    qasm = req.qasm
    max_parts = req.max_parts or MAX_PARTS_DEFAULT
    nodes = list_nodes()
    if not nodes:
        raise HTTPException(status_code=400, detail="No nodes registered")
    available_nodes = [n for n in nodes if n.get('available_qubits',0) > 0]
    if not available_nodes:
        raise HTTPException(status_code=400, detail="No nodes with available qubits")
    target_k = min(len(available_nodes), max_parts)
    if HDH_AVAILABLE:
        try:
            hdh_result = analyze_and_partition_with_hdh(qasm, k=target_k)
        except Exception as e:
            logger.exception("HDH failed, falling back to simple partition")
            hdh_result = fallback_partition(qasm, max_slices=target_k)
    else:
        hdh_result = fallback_partition(qasm, max_slices=target_k)
    slices = hdh_result.get('slices', [])
    assignments = place_slices(slices, nodes)
    cross_pairs = []
    for a in assignments:
        for dep in a.dependencies:
            dep_assign = next((x for x in assignments if x.slice_id==dep), None)
            if dep_assign and dep_assign.assigned_node and a.assigned_node and dep_assign.assigned_node != a.assigned_node:
                cross_pairs.append({"from": dep_assign.assigned_node, "to": a.assigned_node, "count": 1})
    reservations = reserve_epr_pairs(cross_pairs)
    results = []
    slices_results_for_merge = []
    for a in assignments:
        res = send_slice_to_node(a)
        results.append(res)
        if res.get('status') == 'ok' and isinstance(res.get('response'), dict):
            r = res['response']
            slice_result = {
                'slice_id': a.slice_id,
                'logical_qubits': a.logical_qubits,
                'counts': r.get('counts'),
                'measurements': r.get('measurements'),
                'corrections': r.get('corrections', []),
                'execution_time_s': r.get('execution_time_s') or r.get('execution_time_s', None),
                'estimated_fidelity': r.get('estimated_fidelity')
            }
            slices_results_for_merge.append(slice_result)
    total_qubits = extract_total_qubits(qasm)
    merged = None
    if MERGE_AVAILABLE and slices_results_for_merge:
        try:
            merged = merge_all_slices(slices_results_for_merge, total_qubits)
        except Exception as e:
            logger.exception("Merge failed")
            merged = None
    return {"status": "submitted", "job_id": hdh_result.get('job_id'), "assignments": [s.dict() for s in assignments], "epr_reservations": reservations, "node_results": results, "merged_result": merged}


@app.get("/nodes")
def api_nodes():
    return {"nodes": list_nodes()}
