# qnodes/photonic_qnode.py
"""
Photonic QNode (FastAPI REST module)

Salvar em: qnodes/photonic_qnode.py

Endpoints:
 - GET  /health
 - POST /execute_slice

Requisitos:
 pip install fastapi uvicorn requests pydantic

Variáveis de ambiente (opcionais):
 ORCHESTRATOR_URL (default: http://localhost:8000)
 NODE_ID (default: qnode-photonic-1)
 NUM_QUBITS (default: 10)
 API_PORT (default: 9301)
 PHOTON_LOSS_RATE (default: 0.15)
 PHOTON_BASE_LATENCY (default: 0.02)
"""

import os
import time
import random
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

# Config
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
NODE_ID = os.getenv("NODE_ID", "qnode-photonic-1")
NUM_QUBITS = int(os.getenv("NUM_QUBITS", "10"))
API_PORT = int(os.getenv("API_PORT", "9301"))
PHOTON_LOSS_RATE = float(os.getenv("PHOTON_LOSS_RATE", "0.15"))
PHOTON_BASE_LATENCY = float(os.getenv("PHOTON_BASE_LATENCY", "0.02"))
PHOTON_PROB_GATE_FAIL = float(os.getenv("PHOTON_PROB_GATE_FAIL", "0.1"))  # optional per-gate probabilistic fail

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qnode.photonic")

# Pydantic model expected by /execute_slice
class SliceRequest(BaseModel):
    slice_id: str
    qasm: str
    logical_qubits: List[int]
    requirements: Optional[Dict[str, Any]] = {}
    dependencies: Optional[List[str]] = []

# Photonic QNode implementation
class PhotonicQNode:
    def __init__(self, node_id: str, num_qubits: int, api_url: str):
        self.node_id = node_id
        self.num_qubits = num_qubits
        self.api_url = api_url
        self.loss_rate = PHOTON_LOSS_RATE
        self.base_latency = PHOTON_BASE_LATENCY
        # Optionally simulate probabilistic gate failures
        self.prob_gate_fail = PHOTON_PROB_GATE_FAIL

    def register_payload(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "technology": "photonic",
            "num_qubits": self.num_qubits,
            "native_gates": ["bs", "phshift", "cphase"],
            "api_url": self.api_url,
        }

    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "node_id": self.node_id, "num_qubits": self.num_qubits}

    def _simulate_photon_loss(self) -> bool:
        """Return True if a loss event occurs (failure)."""
        return random.random() < self.loss_rate

    def _simulate_gate_fail(self) -> bool:
        """Return True if probabilistic gate failure occurs."""
        return random.random() < self.prob_gate_fail

    def execute_slice(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # quick resource check
        if len(payload.get("logical_qubits", [])) > self.num_qubits:
            return {"status": "error", "reason": "not_enough_qubits", "slice_id": payload.get("slice_id")}

        slice_id = payload.get("slice_id")
        logical_qubits = payload.get("logical_qubits", [])

        # 1) simulate photon loss as a per-slice event
        if self._simulate_photon_loss():
            time.sleep(self.base_latency)
            logger.info("Photonic node: photon loss for slice %s", slice_id)
            return {"status": "error", "reason": "photon_loss", "slice_id": slice_id}

        # 2) small delay proportional to number of qubits (and some randomness)
        exec_delay = self.base_latency + 0.01 * len(logical_qubits) + random.uniform(0, 0.01)
        time.sleep(exec_delay)

        # 3) simulate probabilistic gate failures: if present, return partial-fail
        if self._simulate_gate_fail():
            # we simulate a failed entangling gate — return an error to orchestrator
            logger.info("Photonic node: probabilistic gate failure for slice %s", slice_id)
            return {"status": "error", "reason": "gate_probabilistic_failure", "slice_id": slice_id}

        # 4) produce single-shot measurements per logical qubit (photonic commonly yields probabilistic outcomes)
        measurements = {str(q): random.randint(0, 1) for q in logical_qubits}

        # 5) estimate a fidelity metric (simple heuristic)
        estimated_fidelity = max(0.0, 1.0 - self.loss_rate - 0.01 * len(logical_qubits))

        return {
            "status": "completed",
            "slice_id": slice_id,
            "measurements": measurements,
            "execution_time_s": exec_delay,
            "estimated_fidelity": estimated_fidelity,
        }

# FastAPI app and lifecycle
app = FastAPI(title="Photonic QNode")
_node: Optional[PhotonicQNode] = None

@app.on_event("startup")
def _startup():
    global _node
    api_url = f"http://localhost:{API_PORT}"
    _node = PhotonicQNode(NODE_ID, NUM_QUBITS, api_url)
    try:
        r = requests.post(f"{ORCHESTRATOR_URL}/register", json=_node.register_payload(), timeout=5)
        logger.info("Registered photonic node with orchestrator (status=%s)", getattr(r, "status_code", None))
    except Exception as e:
        logger.warning("Photonic registration failed: %s", e)

@app.get("/health")
def health():
    if _node:
        return _node.health()
    return {"status": "error", "reason": "not_initialized"}

@app.post("/execute_slice")
def execute_slice(req: SliceRequest):
    if _node is None:
        raise HTTPException(status_code=500, detail="node not initialized")
    return _node.execute_slice(req.dict())

# allow running via `python qnodes/photonic_qnode.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("qnodes.photonic_qnode:app", host="0.0.0.0", port=API_PORT, reload=True)
