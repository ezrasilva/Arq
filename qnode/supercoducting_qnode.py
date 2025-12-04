# qnodes/superconducting_qnode.py
"""
Superconducting QNode (FastAPI REST module)

Salvar em: qnodes/superconducting_qnode.py

Endpoints:
 - GET  /health
 - POST /execute_slice

Requisitos:
 pip install qiskit==2.2.3 qiskit-aer fastapi uvicorn requests pydantic

Ambiente (opcionais):
 ORCHESTRATOR_URL (default: http://localhost:8000)
 NODE_ID (default: qnode-superconducting-1)
 NUM_QUBITS (default: 16)
 API_PORT (default: 9101)
 SHOTS (default: 1024)
 SC_SINGLE_Q_ERR, SC_TWO_Q_ERR (ruído usado na heurística de fidelidade)

Uso:
 uvicorn qnodes.superconducting_qnode:app --reload --port 9101
"""

import os
import time
import logging
import re
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

from qiskit import QuantumCircuit, transpile

from qiskit_aer import AerSimulator  # preferred for newer packaging

from qiskit_aer.noise import NoiseModel, depolarizing_error
from qiskit.qasm2.exceptions import QASM2ParseError

# Configuration via env
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
NODE_ID = os.getenv("NODE_ID", "qnode-superconducting-1")
NUM_QUBITS = int(os.getenv("NUM_QUBITS", "16"))
API_PORT = int(os.getenv("API_PORT", "9101"))
SHOTS = int(os.getenv("SHOTS", "1024"))
SC_SINGLE_Q_ERR = float(os.getenv("SC_SINGLE_Q_ERR", "0.001"))
SC_TWO_Q_ERR = float(os.getenv("SC_TWO_Q_ERR", "0.01"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qnode.superconducting")

# Pydantic model expected by /execute_slice
class SliceRequest(BaseModel):
    slice_id: str
    qasm: str
    logical_qubits: List[int]
    requirements: Optional[Dict[str, Any]] = {}
    dependencies: Optional[List[str]] = []

# QNode implementation (encapsulated for testability)
class SuperconductingQNode:
    def __init__(self, node_id: str, num_qubits: int, api_url: str):
        self.node_id = node_id
        self.num_qubits = num_qubits
        self.api_url = api_url
        self.shots = SHOTS
        self.single_q_err = SC_SINGLE_Q_ERR
        self.two_q_err = SC_TWO_Q_ERR
        # instantiate backend lazily (avoid heavy import issues on module import)
        self._backend = None

    @property
    def backend(self):
        if self._backend is None:
            self._backend = AerSimulator()
        return self._backend

    def register_payload(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "technology": "superconducting",
            "num_qubits": self.num_qubits,
            "native_gates": ["x", "y", "z", "h", "cx", "rz"],
            "api_url": self.api_url,
            # --- ADICIONE ISTO ---
            "avg_gate_fidelity": 1.0 - self.two_q_err, # Baseado no erro configurado
            "avg_gate_time_ns": 50.0, # Muito rápido
            "t1_coherence_ns": 80000.0 # ~80us
            # ---------------------
        }

    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "node_id": self.node_id, "num_qubits": self.num_qubits}

    def _make_noise_model(self) -> NoiseModel:
        nm = NoiseModel()
        err1 = depolarizing_error(self.single_q_err, 1)
        err2 = depolarizing_error(self.two_q_err, 2)
        nm.add_all_qubit_quantum_error(err1, ["u1", "u2", "u3", "x", "y", "z", "h", "rx", "ry", "rz"])
        nm.add_all_qubit_quantum_error(err2, ["cx", "cz", "swap"])
        return nm

    def execute_slice(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # basic resource check
        if len(payload.get("logical_qubits", [])) > self.num_qubits:
            return {"status": "error", "reason": "not_enough_qubits", "slice_id": payload.get("slice_id")}

        qasm = payload.get("qasm", "")
        try:
            qc = QuantumCircuit.from_qasm_str(qasm)
        except Exception as e:
            logger.exception("Failed to parse QASM")
            # fallback: many OpenQASM snippets rely on `qelib1.inc` to define gates like h,x,y,z.
            # qiskit's parser may not resolve include paths; try to preprocess the QASM replacing
            # common gates with built-in `u1/u2/u3` forms and remove include directives.
            msg = str(e)
            if isinstance(e, QASM2ParseError) or "cannot use non-builtin custom instruction" in msg:
                try:
                    qasm2 = re.sub(r'include\s+"qelib1.inc"\s*;', '', qasm)
                    # replace usage of common gates with equivalent u-gates
                    replacements = {
                        r'\bh\s+([^;\s]+)': r'u2(0,pi) \1',
                        r'\bx\s+([^;\s]+)': r'u3(pi,0,pi) \1',
                        r'\by\s+([^;\s]+)': r'u3(pi/2,pi/2,pi/2) \1',
                        r'\bz\s+([^;\s]+)': r'u1(pi) \1',
                    }
                    for pat, rep in replacements.items():
                        qasm2 = re.sub(pat, rep, qasm2)
                    qc = QuantumCircuit.from_qasm_str(qasm2)
                    logger.info("QASM parsed after fallback preprocessing")
                except Exception as e2:
                    logger.exception("Fallback QASM parsing failed")
                    return {"status": "error", "reason": f"invalid_qasm: {e2}", "slice_id": payload.get("slice_id")}
            else:
                return {"status": "error", "reason": f"invalid_qasm: {e}", "slice_id": payload.get("slice_id")}

        # transpile + execute with noise model
        start = time.time()
        try:
            transpiled = transpile(qc, backend=self.backend)
            nm = self._make_noise_model()
            job = self.backend.run(transpiled, shots=self.shots, noise_model=nm)
            result = job.result()
            counts = result.get_counts()
        except Exception as e:
            logger.exception("Execution failed on superconducting node")
            return {"status": "error", "reason": f"execution_failed: {e}", "slice_id": payload.get("slice_id")}
        exec_time = time.time() - start

        # naive estimated fidelity heuristic (use number of qubits and CX count)
        estimated_fidelity = max(
            0.0,
            1.0 - (self.single_q_err * qc.num_qubits + self.two_q_err * qc.count_ops().get("cx", 0)),
        )

        return {
            "status": "completed",
            "slice_id": payload.get("slice_id"),
            "counts": counts,
            "execution_time_s": exec_time,
            "estimated_fidelity": estimated_fidelity,
        }

# FastAPI app and lifecycle
app = FastAPI(title="Superconducting QNode")
_node: Optional[SuperconductingQNode] = None

@app.on_event("startup")
def _startup():
    global _node
    api_url = f"http://localhost:{API_PORT}"
    _node = SuperconductingQNode(NODE_ID, NUM_QUBITS, api_url)
    # attempt registration (best-effort)
    try:
        r = requests.post(f"{ORCHESTRATOR_URL}/register", json=_node.register_payload(), timeout=5)
        logger.info("Registered with orchestrator (status=%s)", r.status_code)
    except Exception as e:
        logger.warning("Failed to register with orchestrator: %s", e)

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

# allow running via `python superconducting_qnode.py` for quick dev tests
if __name__ == "__main__":
    import uvicorn
    # módulo correspondente ao caminho atual do arquivo
    uvicorn.run("qnode.supercoducting_qnode:app", host="0.0.0.0", port=API_PORT, reload=True)
