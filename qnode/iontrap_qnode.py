# qnodes/iontrap_qnode.py
"""
IonTrap QNode (FastAPI REST module)

Salvar em: qnodes/iontrap_qnode.py

Requisitos:
pip install qiskit==2.2.3 qiskit-aer fastapi uvicorn requests pydantic

Endpoints:
 - GET  /health
 - POST /execute_slice

Variáveis de ambiente (opcionais):
 ORCHESTRATOR_URL (default: http://localhost:8000)
 NODE_ID (default: qnode-iontrap-1)
 NUM_QUBITS (default: 12)
 API_PORT (default: 9201)
 SHOTS (default: 1024)
 ION_SINGLE_Q_ERR, ION_TWO_Q_ERR (ruído)
 ION_BASE_LATENCY (default: 0.05)
"""

import os
import time
import logging
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import requests

# Qiskit imports (qiskit 2.2.3)
from qiskit import QuantumCircuit, transpile

# prefer qiskit_aer package if present
from qiskit_aer import AerSimulator  # type: ignore



from qiskit_aer.noise import NoiseModel, depolarizing_error


# Config from env
ORCHESTRATOR_URL = os.getenv("ORCHESTRATOR_URL", "http://localhost:8000")
NODE_ID = os.getenv("NODE_ID", "qnode-iontrap-1")
NUM_QUBITS = int(os.getenv("NUM_QUBITS", "12"))
API_PORT = int(os.getenv("API_PORT", "9201"))
SHOTS = int(os.getenv("SHOTS", "1024"))
ION_SINGLE_Q_ERR = float(os.getenv("ION_SINGLE_Q_ERR", "0.0005"))
ION_TWO_Q_ERR = float(os.getenv("ION_TWO_Q_ERR", "0.005"))
ION_BASE_LATENCY = float(os.getenv("ION_BASE_LATENCY", "0.05"))

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("qnode.iontrap")


def _ensure_qelib_include(qasm: str) -> str:
    """Garantir que o QASM inclui qelib1.inc para que gates built-in sejam reconhecidos.
    Se a inclusão estiver ausente, insere `include "qelib1.inc";` após a linha de versão
    `OPENQASM 2.0;`. Se não houver declaração de versão, insere um cabeçalho padrão.
    """
    if not qasm or "qelib1.inc" in qasm:
        return qasm

    lines = qasm.splitlines()
    for i, line in enumerate(lines):
        if line.strip().lower().startswith("openqasm"):
            # if the version line contains additional statements after the semicolon
            # (e.g. "OPENQASM 2.0; qreg q[2]; h q[0];"), split them so the include
            # is placed before any subsequent statements.
            rest = ""
            if ";" in line:
                parts = line.split(";", 1)
                version_part = parts[0].strip() + ";"
                rest = parts[1].lstrip()
                lines[i] = version_part
                insert_at = i + 1
                lines.insert(insert_at, 'include "qelib1.inc";')
                if rest:
                    # place the remainder as a new line after the include
                    lines.insert(insert_at + 1, rest)
            else:
                lines.insert(i + 1, 'include "qelib1.inc";')
            return "\n".join(lines)

    header = 'OPENQASM 2.0;\ninclude "qelib1.inc";'
    return header + "\n" + qasm

# Pydantic model
class SliceRequest(BaseModel):
    slice_id: str
    qasm: str
    logical_qubits: List[int]
    requirements: Optional[Dict[str, Any]] = {}
    dependencies: Optional[List[str]] = []

# IonTrap QNode implementation
class IonTrapQNode:
    def __init__(self, node_id: str, num_qubits: int, api_url: str):
        self.node_id = node_id
        self.num_qubits = num_qubits
        self.api_url = api_url
        self.shots = SHOTS
        self.single_q_err = ION_SINGLE_Q_ERR
        self.two_q_err = ION_TWO_Q_ERR
        self.base_latency = ION_BASE_LATENCY
        self._backend = None

    @property
    def backend(self):
        if self._backend is None:
            # instantiate AerSimulator lazily
            self._backend = AerSimulator()
        return self._backend

    def register_payload(self) -> Dict[str, Any]:
        return {
            "id": self.node_id,
            "technology": "ion_trap",
            "num_qubits": self.num_qubits,
            "native_gates": ["ms", "rx", "ry", "rz"],
            "api_url": self.api_url,
            "avg_gate_fidelity": 1.0 - self.two_q_err,
            "avg_gate_time_ns": 100000.0, # Lento (100us)
            "t1_coherence_ns": 1000000000.0 # T1 Gigante
        }

    def health(self) -> Dict[str, Any]:
        return {"status": "ok", "node_id": self.node_id, "num_qubits": self.num_qubits}

    def _make_noise_model(self):
        # handle gracefully if noise utilities unavailable
        if NoiseModel is None or depolarizing_error is None:
            return None
        nm = NoiseModel()
        err1 = depolarizing_error(self.single_q_err, 1)
        err2 = depolarizing_error(self.two_q_err, 2)
        nm.add_all_qubit_quantum_error(err1, ["u1", "u2", "u3", "x", "y", "z", "h", "rx", "ry", "rz"])
        nm.add_all_qubit_quantum_error(err2, ["cx", "cz", "swap"])
        return nm

    def _map_ms_gates(self, qc: "QuantumCircuit") -> "QuantumCircuit":
        """Heuristic mapping of custom 'ms' gate to CX + single-qubit rotations.
        If input circuit has no 'ms' instructions, returns the original circuit.
        """
        try:
            # quick check for 'ms' presence
            has_ms = any(getattr(instr, "name", "") == "ms" for instr, _, _ in qc.data)
            if not has_ms:
                return qc

            new_qc = QuantumCircuit(qc.num_qubits)
            for instr, qargs, cargs in qc.data:
                name = getattr(instr, "name", None)
                if name == "ms" and len(qargs) >= 2:
                    q0 = qargs[0].index
                    q1 = qargs[1].index
                    # heuristic decomposition: CX + small RZs
                    new_qc.cx(q0, q1)
                    new_qc.rz(0.5, q0)
                    new_qc.rz(0.5, q1)
                else:
                    # preserve other instructions
                    try:
                        new_qc.append(instr, [q.index for q in qargs], [c.index for c in cargs] if cargs else [])
                    except Exception:
                        # fallback: append instruction by name (best-effort)
                        # if cannot append, ignore to avoid breaking pipeline
                        pass
            return new_qc
        except Exception:
            # if mapping fails for any reason, return original to avoid breaking pipeline
            return qc

    def execute_slice(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        # resource check
        if len(payload.get("logical_qubits", [])) > self.num_qubits:
            return {"status": "error", "reason": "not_enough_qubits", "slice_id": payload.get("slice_id")}

        qasm = payload.get("qasm", "")
        # ensure commonly-missing qelib include is present so built-in gates parse
        qasm = _ensure_qelib_include(qasm)
        try:
            qc = QuantumCircuit.from_qasm_str(qasm)
        except Exception as e:
            logger.exception("Failed to parse QASM")
            return {"status": "error", "reason": f"invalid_qasm: {e}", "slice_id": payload.get("slice_id")}

        # map ms gates -> decomposed sequence
        qc_mapped = self._map_ms_gates(qc)

        # simulate higher latency characteristic of ion trap gates
        time.sleep(self.base_latency + 0.01 * qc_mapped.size())

        # if circuit has no measurements, add measurements to all qubits
        if qc_mapped.count_ops().get("measure", 0) == 0:
            logger.info("No measurements found in circuit — adding measure_all()")
            try:
                qc_mapped.measure_all()
            except Exception:
                # fallback: explicitly add a classical register and measure each qubit
                from qiskit import ClassicalRegister
                creg = ClassicalRegister(qc_mapped.num_qubits)
                qc_mapped.add_register(creg)
                for i in range(qc_mapped.num_qubits):
                    qc_mapped.measure(i, i)

        # execute
        start = time.time()
        try:
            transpiled = transpile(qc_mapped, backend=self.backend)
            nm = self._make_noise_model()
            if nm is not None:
                job = self.backend.run(transpiled, shots=self.shots, noise_model=nm)
            else:
                job = self.backend.run(transpiled, shots=self.shots)
            result = job.result()
            try:
                counts = result.get_counts()
            except Exception as e:
                logger.warning("No counts in result, attempting to use memory: %s", e)
                try:
                    memory = result.get_memory()
                    from collections import Counter
                    counts = dict(Counter(memory))
                except Exception:
                    counts = {}
        except Exception as e:
            logger.exception("Execution failed on iontrap node")
            return {"status": "error", "reason": f"execution_failed: {e}", "slice_id": payload.get("slice_id")}
        exec_time = time.time() - start

        estimated_fidelity = max(0.0, 1.0 - (self.single_q_err * qc_mapped.size() + self.two_q_err * qc_mapped.count_ops().get("cx", 0)))

        # retorno formatado conforme solicitado: campos essenciais + métricas
        return {
            "status": "completed",
            "slice_id": payload.get("slice_id"),
            "node_id": self.node_id,
            "counts": counts,
            "execution_time_s": exec_time,
            "estimated_fidelity": estimated_fidelity,
        }

# FastAPI app & lifecycle
app = FastAPI(title="IonTrap QNode")
_node: Optional[IonTrapQNode] = None

@app.on_event("startup")
def _startup():
    global _node
    api_url = f"http://localhost:{API_PORT}"
    _node = IonTrapQNode(NODE_ID, NUM_QUBITS, api_url)
    # best-effort registration
    try:
        r = requests.post(f"{ORCHESTRATOR_URL}/register", json=_node.register_payload(), timeout=5)
        logger.info("Registered IonTrap node with orchestrator (status=%s)", getattr(r, "status_code", None))
    except Exception as e:
        logger.warning("IonTrap registration failed: %s", e)

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

# allow running as `python qnodes/iontrap_qnode.py`
if __name__ == "__main__":
    import uvicorn
    uvicorn.run("qnodes.iontrap_qnode:app", host="0.0.0.0", port=API_PORT, reload=True)
