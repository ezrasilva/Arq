"""
orchestrator_merge.py

Utilities to merge results from distributed slice execution into a single logical
result for the user. This is a pragmatic MVP implementation suitable for the
SBRC submission: it assumes slices measure disjoint sets of logical qubits (the
usual output-of-circuit case) and merges per-slice `counts` or `measurements`.

Provided functions:
 - apply_classical_corrections_to_bitstring(bitstr, corrections, qubit_positions)
 - expand_slice_bitstring_to_full(bitstr, slice_qubits, total_qubits)
 - merge_two_counts(counts_a, bits_a, counts_b, bits_b, total_qubits)
 - merge_all_slices(slices_results, total_qubits)
 - compute_overall_estimated_fidelity(slices_results)

Schema expectations for each slice result (dict):
{
  "slice_id": "s0",
  "logical_qubits": [0,1],            # global positions
  "counts": {"00": 512, "11": 512}, # optional OR
  "measurements": {"0": 0, "1": 1}, # optional (single-shot style)
  "corrections": [{"target_qubit": 1, "type": "X"}],
  "execution_time_s": 0.12,
  "estimated_fidelity": 0.92
}

Notes & approximations:
- When only `counts` are provided per slice, we assume independence between
  slices and build a joint probability distribution by taking the outer-product
  of per-slice probability vectors (common practical approximation for MVP).
- `measurements` (single-shot outcomes) are merged deterministically.
- Classical corrections are applied before merging.

Example usage:
  final = merge_all_slices(slices_results, total_qubits=9)

"""

from collections import Counter
from typing import Dict, List, Any, Tuple
import math

# ----------------- helpers -----------------

def apply_classical_corrections_to_bitstring(bitstr: str, corrections: List[Dict[str, Any]], qubit_positions: List[int]) -> str:
    """Apply classical Pauli corrections (X/Z) to a bitstring representing
    only the slice's bits.

    - bitstr: e.g. '01' (MSB..LSB convention used consistently across system)
    - corrections: list of {"target_qubit": global_index, "type": "X"/"Z"}
    - qubit_positions: list mapping bit positions in bitstr -> global qubit indices

    Returns corrected bitstring (same length as input).
    For MVP we only implement X corrections (bit flips). Z corrections are
    ignored at measurement-level (they affect phase, not classical bit outcome).
    """
    if not corrections:
        return bitstr
    # convert to list for mutability; assume bitstr[0] is MSB (leftmost)
    bits = list(bitstr)
    # map global target -> local index in bitstr
    pos_to_local = {g: i for i, g in enumerate(qubit_positions)}
    for corr in corrections:
        t = corr.get('type', '').upper()
        gq = corr.get('target_qubit')
        if t == 'X' and gq in pos_to_local:
            li = pos_to_local[gq]
            # flip bit
            bits[li] = '1' if bits[li] == '0' else '0'
    return ''.join(bits)


def expand_slice_bitstring_to_full(bitstr: str, slice_qubits: List[int], total_qubits: int) -> str:
    """Expand a bitstring that refers to `slice_qubits` (global indices) into
    a full-length bitstring of size total_qubits.

    Convention: both slice bitstrings and returned full bitstrings are MSB..LSB
    (index 0 -> leftmost) and slice_qubits are ordered consistently with that.
    """
    full = ['0'] * total_qubits
    # assume bitstr length equals len(slice_qubits)
    for i, gq in enumerate(slice_qubits):
        if i >= len(bitstr):
            break
        full[gq] = bitstr[i]
    return ''.join(full)

# ----------------- merging counts -----------------

def _counts_to_prob_vec(counts: Dict[str, int]) -> Tuple[List[str], List[float]]:
    """Convert counts dict to ordered list of bitstrings and corresponding
    probability vector (normalized).
    Returns (bitstrings_list, prob_list)
    """
    total = sum(counts.values())
    if total == 0:
        return (list(counts.keys()), [0.0]*len(counts))
    keys = list(counts.keys())
    probs = [counts[k] / total for k in keys]
    return keys, probs


def merge_two_counts(counts_a: Dict[str,int], bits_a: List[int], counts_b: Dict[str,int], bits_b: List[int], total_qubits: int) -> Dict[str,int]:
    """Merge two slices' counts into a joint counts dict over total_qubits.
    Approach:
      - convert counts_a -> probs over its strings
      - convert counts_b -> probs
      - joint probs = outer product of probs (assumes independence)
      - rebuild full bitstrings by placing bits of a and b into full positions
      - return counts scaled to common_shots (use LCM of shot counts or scale to floats then round)
    """
    keys_a, probs_a = _counts_to_prob_vec(counts_a)
    keys_b, probs_b = _counts_to_prob_vec(counts_b)
    shots_a = sum(counts_a.values()) if counts_a else 0
    shots_b = sum(counts_b.values()) if counts_b else 0
    # target shots: use geometric mean to keep numbers reasonable if unequal
    target_shots = int(math.sqrt(max(1, shots_a) * max(1, shots_b)))

    joint_counter: Counter = Counter()
    for sa, pa in zip(keys_a, probs_a):
        for sb, pb in zip(keys_b, probs_b):
            joint_prob = pa * pb
            # build full bitstring
            full = ['0'] * total_qubits
            # place bits from sa into bits_a positions
            for ia, gq in enumerate(bits_a):
                full[gq] = sa[ia] if ia < len(sa) else '0'
            for ib, gq in enumerate(bits_b):
                full[gq] = sb[ib] if ib < len(sb) else '0'
            fulls = ''.join(full)
            joint_count = joint_prob * target_shots
            joint_counter[fulls] += joint_count
    # convert to int counts by rounding; ensure total matches target_shots by normalization
    total_est = sum(joint_counter.values())
    if total_est == 0:
        return {}
    # normalize to integer counts summing to target_shots
    scaled = {k: int(round(v * (target_shots / total_est))) for k, v in joint_counter.items()}
    # adjust rounding error
    diff = target_shots - sum(scaled.values())
    if diff != 0 and len(scaled) > 0:
        # add/subtract diff to the largest entry
        kmax = max(scaled.items(), key=lambda x: x[1])[0]
        scaled[kmax] += diff
    return scaled


def merge_all_slices(slices_results: List[Dict[str, Any]], total_qubits: int) -> Dict[str, Any]:
    """Main function to merge multiple slice results.

    Each slice_result must include: slice_id, logical_qubits (list), and either
    `counts` or `measurements`. Corrections are applied if present.

    Returns a dict:
    {
      "counts": {full_bitstring: count, ...},
      "total_shots": N,
      "fidelity_estimate": 0.92,
      "metadata": { per-slice metadata }
    }
    """
    # sort slices by slice_id for deterministic behavior
    slices_sorted = sorted(slices_results, key=lambda s: s.get('slice_id',''))

    # normalize per-slice: apply corrections to keys of counts or to measurement mapping
    normalized = []
    for s in slices_sorted:
        slice_qubits = s.get('logical_qubits', [])
        corrs = s.get('corrections', []) or []
        if 'counts' in s and s['counts']:
            # apply corrections to each bitstring key
            new_counts = {}
            for k, v in s['counts'].items():
                corrected = apply_classical_corrections_to_bitstring(k, corrs, slice_qubits)
                new_counts[corrected] = new_counts.get(corrected, 0) + v
            normalized.append({'counts': new_counts, 'logical_qubits': slice_qubits, 'slice_id': s.get('slice_id')})
        elif 'measurements' in s and s['measurements']:
            # single-shot style measurements -> convert to counts of 1
            # measurements mapping is qubit->bit
            # build bitstring in order of slice_qubits
            bits = ''.join(str(s['measurements'].get(str(q),0)) for q in slice_qubits)
            bits = apply_classical_corrections_to_bitstring(bits, corrs, slice_qubits)
            normalized.append({'counts': {bits: 1}, 'logical_qubits': slice_qubits, 'slice_id': s.get('slice_id')})
        else:
            # no measurement data, skip
            normalized.append({'counts': {}, 'logical_qubits': slice_qubits, 'slice_id': s.get('slice_id')})

    # progressively merge counts
    if not normalized:
        return {'counts': {}, 'total_shots': 0, 'fidelity_estimate': 0.0, 'metadata': {}}

    merged = normalized[0]['counts']
    merged_bits = normalized[0]['logical_qubits']
    for nxt in normalized[1:]:
        merged = merge_two_counts(merged, merged_bits, nxt['counts'], nxt['logical_qubits'], total_qubits)
        # after merging, the merged now represents full-bitstrings in total_qubits space
        # subsequent merges must treat merged as full-length counts. So update merged_bits to full range
        merged_bits = list(range(total_qubits))

    total_shots = sum(merged.values()) if merged else 0
    fidelity = compute_overall_estimated_fidelity(slices_results)

    return {
        'counts': merged,
        'total_shots': total_shots,
        'fidelity_estimate': fidelity,
        'metadata': {s.get('slice_id'): { 'execution_time_s': s.get('execution_time_s'), 'estimated_fidelity': s.get('estimated_fidelity') } for s in slices_results}
    }

# ----------------- fidelity estimation -----------------

def compute_overall_estimated_fidelity(slices_results: List[Dict[str, Any]]) -> float:
    """Simple composition of per-slice estimated fidelities: multiply them.
    This is a simplifying assumption (independence); we return product clipped to [0,1].
    """
    prod = 1.0
    any_val = False
    for s in slices_results:
        f = s.get('estimated_fidelity')
        if f is not None:
            prod *= max(0.0, min(1.0, float(f)))
            any_val = True
    if not any_val:
        return 0.0
    return max(0.0, min(1.0, prod))

# ----------------- end -----------------
