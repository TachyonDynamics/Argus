# Argus

**Local AI monitor for Ollama** — transparent proxy, request diagnostics, and model integrity checking.

Argus sits in front of your local Ollama instance. It records every request, breaks down where time is spent (load, prompt, generation, queue, etc.), and can verify that your model blobs have not been corrupted or swapped.

---

## Features

- **Transparent proxy** — point your tools at Argus instead of Ollama; everything just works
- **Request diagnostics** — clear timing breakdown and plain-English explanations of slowdowns
- **Model integrity** — SHA-256 verification of Ollama blobs + manifest change detection
- **Local-only admin UI** — view live status, request history, and integrity results (bound to `127.0.0.1`)
- **Zero cloud** — everything stays on your machine

---

## Requirements

- Python 3.11+
- A running [Ollama](https://ollama.com) instance

---

## Quick Install (from GitHub)

```bash
pip install "git+https://github.com/TachyonDynamics/Argus.git"
```

Or with `uv` / `pipx`:

```bash
uv pip install "git+https://github.com/TachyonDynamics/Argus.git"
# or
pipx install "git+https://github.com/TachyonDynamics/Argus.git"
```

---

## Usage

1. Start Ollama as usual (default port 11434).

2. Run Argus:

```bash
argus
```

By default Argus:
- Listens for client connections on port **11434**
- Forwards to Ollama on **11435** (or the port you configure)
- Serves the admin UI on **47800** (localhost only)

3. Point your applications / Open WebUI / scripts at `http://127.0.0.1:11434` as usual.

4. Open the admin UI:

```
http://127.0.0.1:47800
```

### Useful flags

```bash
argus --help

argus --upstream http://127.0.0.1:11435
argus --proxy-port 11434
argus --admin-port 47800
argus --data-dir ~/.argus
argus --verbose
```

---

## What the UI shows

- **Now** — current live activity
- **Requests** — history of all proxied calls with timing breakdowns
- **Integrity** — results of model blob / manifest scans (detect corruption or unexpected model changes)

---

## Development

```bash
git clone https://github.com/TachyonDynamics/Argus.git
cd Argus
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
pytest
```

---

## License

GNU General Public License v3.0 or later (GPL-3.0-or-later).

---

## Company

Developed by [Tachyon Dynamics](https://www.tachyondynamics.ca)  
GitHub: [github.com/TachyonDynamics](https://github.com/TachyonDynamics)

Support / contact: support@tachyondynamics.ca
