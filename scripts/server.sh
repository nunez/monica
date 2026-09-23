#!/bin/zsh
# monica server control: ./scripts/server.sh {start|stop|restart|status|log}
set -e
ROOT="$(cd "$(dirname "$0")/.." && pwd)"
HOST="${MONICA_HOST:-0.0.0.0}"
PORT="${MONICA_PORT:-8910}"
PIDFILE="$ROOT/server.pid"
LOG="$ROOT/server.log"

running() { [[ -f "$PIDFILE" ]] && kill -0 "$(cat "$PIDFILE")" 2>/dev/null; }

case "${1:-status}" in
  start)
    if running; then echo "already running (pid $(cat "$PIDFILE"))"; exit 0; fi
    cd "$ROOT"
    nohup env PYTHONPATH=src .venv/bin/python -m uvicorn monica.server:app \
      --host "$HOST" --port "$PORT" >>"$LOG" 2>&1 &
    echo $! > "$PIDFILE"
    echo "started pid $(cat "$PIDFILE") on http://$HOST:$PORT (log: $LOG)"
    ;;
  stop)
    if running; then kill "$(cat "$PIDFILE")" && rm -f "$PIDFILE" && echo stopped; else echo "not running"; fi
    ;;
  restart)
    "$0" stop; sleep 1; "$0" start
    ;;
  status)
    if running; then
      echo "running (pid $(cat "$PIDFILE"))"
      curl -s -o /dev/null -w "health: %{http_code} on http://127.0.0.1:$PORT/v1/systemone\n" \
        -X POST "http://127.0.0.1:$PORT/v1/systemone" \
        -H 'content-type: application/json' -d '{"questions":{}}'
    else
      echo "not running"
    fi
    ;;
  log)
    tail -n 40 "$LOG"
    ;;
  *)
    echo "usage: $0 {start|stop|restart|status|log}" >&2
    exit 2
    ;;
esac
