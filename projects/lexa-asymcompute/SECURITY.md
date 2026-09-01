# Security policy

Report security issues privately through GitHub's security-reporting interface when available.

The toolkit parses local JSON and JSONL supplied by the user. It does not execute trace content, download models or connect to external targets. Route parsing is fail-closed for unknown fields by default to reduce accidental disclosure of prompts, generated text, token IDs, logits, embeddings or secrets.
