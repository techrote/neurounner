# Host-repository fit and integration boundary

MEMPLANT is a **bounded research folder in neurounner**, as requested. It does not silently turn neurounner into a generic cache serving platform.

Existing useful infrastructure:
- [README](../README.md): real mixed-model hosted CPU inference and a resident multi-model ring.
- [Conversation topologies](../docs/CONVERSATION_TOPOLOGIES.md): independent panels, alternating model roles, model ecology and controlled handoffs.
- [RAG map](../RAG.md) and [dataset filing](../docs/DATASET_FILING_SYSTEM.md): provenance, authoritative raw records, negative results and reproducibility.
- [Model candidate registry](../config/candidates.json), [model journal](../docs/MODEL_JOURNAL.md) and [model storage](../docs/MODEL_STORAGE.md): pinned source and cost profile.

Neurounner's current paired models may not have geometry/runtime combinations suitable for direct KV insertion. Select an M1 model pair **after** a host/runtime feasibility check, not by assuming Sprocket/LFM caches are already interoperable.

Potential later users: `intrallm` for orchestration, the Training Triumvirate for controlled memory inheritance, and specialized inference engines for restricted-resource cache experiments. All integration remains explicitly out of scope for the documentation increment.

A separate MEMPLANT repository may become appropriate if it grows into a real cross-runtime library or inference-service control plane. Until then, the authoritative documents live only under this subfolder, with no changes to runners, issues or the host project's existing experiment logic.
