from hdh.importers import from_qasm
from hdh.partitioning import partition as hdh_partition
import re
import uuid

def fix_qasm_header(qasm_fragment: str) -> str:
    """
    Conserta um fragmento de QASM que pode ter perdido o cabeçalho 'qreg'.
    1. Remove cabeçalhos antigos ou incorretos.
    2. Descobre qual o maior índice de qubit usado (ex: q[10]).
    3. Adiciona um cabeçalho novo 'qreg q[11];' e 'creg c[11];'.
    """
    # 1. Encontrar todos os usos de q[...]
    qubit_indices = [int(i) for i in re.findall(r"q\[(\d+)\]", qasm_fragment)]
    
    if not qubit_indices:
        return qasm_fragment # Se não tem qubits, retorna como está
        
    max_qubit = max(qubit_indices)
    total_qubits = max_qubit + 1
    
    # 2. Remover linhas de qreg/creg antigas para evitar duplicação
    lines = qasm_fragment.split('\n')
    cleaned_lines = [
        line for line in lines 
        if not line.strip().startswith('qreg') 
        and not line.strip().startswith('creg')
        and not line.strip().startswith('OPENQASM')
        and not line.strip().startswith('include')
    ]
    
    # 3. Montar o novo cabeçalho
    header = [
        "OPENQASM 2.0;",
        'include "qelib1.inc";',
        f"qreg q[{total_qubits}];",
        f"creg c[{total_qubits}];"
    ]
    
    return "\n".join(header + cleaned_lines)



def analyze_and_partition_with_hdh(qasm: str, k: int) -> Dict[str, Any]:
    '''Analisa o circuito QASM e particiona ele usando o HDH que é uma Represetação intermediaria de circuito quanticos que permite particionamento eficiente.
    Args:
        qasm (str): O circuito em formato QASM.
        k (int): Número desejado de partições.
    Returns:
        Retorna um dicionário contendo as fatias do circuito particionado.'''
    graph = from_qasm(qasm)

    for edge in graph.edges:
        if edge.is_complex_gate: 
            edge.weight = 100 # Força o particionador a manter esses juntos
        else:
            edge.weight = 1   # Pode cortar se necessário

    try:
        parts = hdh_partition(graph, k=k)
        subgraphs = getattr(parts, 'subgraphs', None) or getattr(parts, 'parts', None) or parts
        slices = []
        for i, sg in enumerate(subgraphs):
            try:
                raw_qasm = sg.to_qasm()
                qasm_slice = fix_qasm_header(raw_qasm)
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
