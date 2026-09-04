#!/usr/bin/env bash
# Run this and open http://localhost:5500
# ...or override with your own port.
set -euo pipefail

cd "$(dirname "$0")"

PORT="${PORT:-5500}"

echo "Serving Pop Art IRL on http://localhost:${PORT}"
exec python3 -m http.server "$PORT"
