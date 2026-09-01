from lexa_asymcompute.provenance import validate_evidence_payload

record = validate_evidence_payload({
    "classification": "observed_measurement",
    "environment": "lab-node-01",
    "measured": True,
    "notes": "Example only; replace with a real immutable benchmark record.",
})
print(record)
