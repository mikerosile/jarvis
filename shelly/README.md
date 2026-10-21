# S.H.E.L.L.Y. - Shell Handling, Execution, and Local/Linked-system Logistics Yield

SHELLY is the hands-on companion of the J.A.R.V.I.S. stack. It is built on the [Open-Terminal](https://docs.openwebui.com/features/open-terminal/) software from the Open-WebUI suite and provides a sandboxed environment where files can be accessed and a simple terminal is available for code execution and testing.

By giving J.A.R.V.I.S. (the Open-WebUI frontend, run by Loki) direct access to a shell on local or linked systems, SHELLY lets it answer simple questions — like *"What time is it?"* — by running commands locally, without having to reach the Internet. It also serves as a practical workbench: write, edit, and test code in real time instead of relying solely on generated responses.

## Configuration

SHELLY's configuration is provided by [`../compose.yaml`](../compose.yaml) and [`../.env`](../.env). The service definition lives under `services.shelly` in the compose file, while its runtime values are supplied through environment variables loaded from `.env`.

| Item | Value / Source | Description |
| ---- | -------------- | ----------- |
| Image | `ghcr.io/open-webui/open-terminal:0.14.1` | Pinned Open-Terminal release |
| Container name | `jarvis_shelly` | Identifies the container within the stack |
| Port mapping | `8001:8000` | Host port 8001 → container port 8000 |
| Storage volume | `shelly-data:/home/user` | Named volume backing the user home directory and file access |
| Restart policy | `unless-stopped` | Restarts automatically unless manually stopped |

### Environment Variables

The following variables are wired from `../.env` into the container (with safe defaults in `compose.yaml`):

| Variable | Purpose |
| -------- | ------- |
| `OPEN_TERMINAL_API_KEY` | API key required by Open-Terminal clients such as Open-WebUI to access its REST/terminal endpoints. |
| `OPEN_TERMINAL_MULTI_USER` | Enables multi-user mode when set to `true`, allowing more than one user session in the terminal environment. |
| `OPEN_TERMINAL_CORS_ALLOWED_ORIGINS` | Restricts which origins (e.g., the Open-WebUI frontend URL) may call the Open-Terminal API cross-origin. |

## Role in the J.A.R.V.I.S. Stack

- **File access:** the mounted `shelly-data` volume gives SHELLY a persistent user home directory (`/home/user`) that can be browsed and worked on through its file interface.
- **Code execution & testing:** the built-in terminal provides a simple shell for running, debugging, and iterating on code against local or linked systems.
- **Local intelligence:** integrated with Open-WebUI (Loki), SHELLY allows J.A.R.V.I.S. to answer basic questions directly from the host environment — no Internet round-trip required. For example, "What time is it?"

## References

- [Open-Terminal documentation](https://docs.openwebui.com/features/open-terminal/)
- [Open-Terminal GitHub repository](https://github.com/open-webui/open-terminal)
