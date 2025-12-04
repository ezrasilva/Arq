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
from orchestrator.src.partition import analyze_and_partition_with_hdh
from pydantic import BaseModel
from typing import Dict, List, Any, Optional
import threading
import uuid
import logging
import time
import os
import re

import requests

app = FastAPI(title="Orchestrator with Merge - Priority A")

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("orchestrator")

# --- In-memory registry ---
NODES: Dict[str, Dict[str, Any]] = {}
NODES_LOCK = threading.Lock()


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
    nodes = list_nodes()
    if not nodes:
        raise HTTPException(status_code=400, detail="No nodes registered")
    available_nodes = [n for n in nodes if n.get('available_qubits',0) > 0]
    if not available_nodes:
        raise HTTPException(status_code=400, detail="No nodes with available qubits")
    target_k = min(len(available_nodes), max_parts)
  
    hdh_result = analyze_and_partition_with_hdh(qasm, k=target_k)
    
    slices = hdh_result.get('slices', [])
   
    return {"status": "submitted", "job_id": hdh_result.get('job_id'), "assignments": [s.dict() for s in assignments], "epr_reservations": reservations, "node_results": results, "merged_result": merged}


@app.get("/nodes")
def api_nodes():
    return {"nodes": list_nodes()}
