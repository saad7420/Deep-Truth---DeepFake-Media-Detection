#!/usr/bin/env bash
# =============================================================================
# Deep Truth — one-command setup and run (Linux / macOS)
#
#   ./scripts/start.sh            check deps, install what's missing, run it all
#   ./scripts/start.sh --check    report only, change nothing
#   ./scripts/start.sh --stop     stop everything this script started
#   ./scripts/start.sh --restart  stop then start
#   ./scripts/start.sh --logs     follow all logs
#   ./scripts/start.sh --status   what is running right now
#
# Flags:
#   --yes            don't prompt before installing system packages
#   --no-web         skip the Next.js console (API + worker only)
#   --workers N      worker slots (default 2; each loads its own model copies)
#
# The stack is four processes: Redis, the FastAPI API, one or more Celery
# workers, and the Next.js console. Starting them by hand means four terminals
# and remembering that the worker does NOT hot-reload — which is the single
# most common way to spend twenty minutes debugging code that is not running.
# =============================================================================
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SERVER_DIR="$REPO_ROOT/server"
CLIENT_DIR="$REPO_ROOT/client"
RUN_DIR="$REPO_ROOT/.run"

WORKERS=2
ASSUME_YES=0
WITH_WEB=1
MODE="start"

C_RESET=$'\033[0m'; C_DIM=$'\033[2m'; C_RED=$'\033[31m'
C_GREEN=$'\033[32m'; C_YELLOW=$'\033[33m'; C_BLUE=$'\033[34m'; C_BOLD=$'\033[1m'

ok()    { printf "  ${C_GREEN}✓${C_RESET} %s\n" "$*"; }
warn()  { printf "  ${C_YELLOW}!${C_RESET} %s\n" "$*"; }
bad()   { printf "  ${C_RED}✗${C_RESET} %s\n" "$*"; }
info()  { printf "  ${C_DIM}%s${C_RESET}\n" "$*"; }
step()  { printf "\n${C_BOLD}%s${C_RESET}\n" "$*"; }
die()   { printf "\n${C_RED}%s${C_RESET}\n" "$*" >&2; exit 1; }

while [[ $# -gt 0 ]]; do
  case "$1" in
    --check)   MODE="check" ;;
    --stop)    MODE="stop" ;;
    --restart) MODE="restart" ;;
    --logs)    MODE="logs" ;;
    --status)  MODE="status" ;;
    --yes|-y)  ASSUME_YES=1 ;;
    --no-web)  WITH_WEB=0 ;;
    --workers) shift; WORKERS="${1:-2}" ;;
    -h|--help) sed -n '2,25p' "${BASH_SOURCE[0]}" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) die "Unknown option: $1  (try --help)" ;;
  esac
  shift
done

mkdir -p "$RUN_DIR"

# ─── process helpers ─────────────────────────────────────────────────────────
# PID files rather than pkill patterns: pkill would also kill a worker the user
# started by hand in another terminal, which is rude and confusing.

pid_of()      { [[ -f "$RUN_DIR/$1.pid" ]] && cat "$RUN_DIR/$1.pid" || echo ""; }
is_running()  { local p; p="$(pid_of "$1")"; [[ -n "$p" ]] && kill -0 "$p" 2>/dev/null; }

start_bg() {  # start_bg <name> <workdir> <command...>
  local name="$1" dir="$2"; shift 2
  if is_running "$name"; then
    ok "$name already running (pid $(pid_of "$name"))"
    return
  fi

  # `exec` inside the subshell matters: without it the subshell lingers as the
  # child and `$!` is its pid rather than the server's, so the pid file points
  # at a shell that has already gone. `disown` matters just as much — a bare
  # `&` leaves the job in bash's table and the script then blocks on it at
  # exit, which is exactly how the first version of this hung forever with
  # every service up and running.
  ( cd "$dir" && exec "$@" ) >"$RUN_DIR/$name.log" 2>&1 </dev/null &
  local pid=$!
  echo "$pid" >"$RUN_DIR/$name.pid"
  disown "$pid" 2>/dev/null || true

  sleep 1
  if is_running "$name"; then
    ok "$name started (pid $(pid_of "$name"), log .run/$name.log)"
  else
    bad "$name failed to start — last lines of .run/$name.log:"
    tail -n 15 "$RUN_DIR/$name.log" 2>/dev/null | sed 's/^/      /'
    return 1
  fi
}

# Every pid beneath $1, depth-first. `npm run dev` becomes sh -> node, and
# celery's prefork pool is a tree of children — killing only the pid we
# recorded orphans the rest, which then keep holding port 3000 and the queue.
descendants() {
  local parent="$1" child
  for child in $(pgrep -P "$parent" 2>/dev/null); do
    descendants "$child"
    printf '%s ' "$child"
  done
}

stop_one() {
  local name="$1" p; p="$(pid_of "$name")"
  if [[ -z "$p" ]] || ! kill -0 "$p" 2>/dev/null; then
    rm -f "$RUN_DIR/$name.pid"; return
  fi

  # The tree has to be walked *before* the parent dies. Afterwards the children
  # are reparented to init and there is no longer any way to find them from
  # the pid we hold.
  local tree; tree="$(descendants "$p")$p"

  # TERM first so celery can finish the task it is on and uvicorn can close
  # its sockets; KILL only what refuses.
  # shellcheck disable=SC2086
  kill -TERM $tree 2>/dev/null || true

  local remaining
  for _ in $(seq 1 20); do
    remaining=""
    for q in $tree; do kill -0 "$q" 2>/dev/null && remaining+="$q "; done
    [[ -z "$remaining" ]] && break
    sleep 0.5
  done

  if [[ -n "$remaining" ]]; then
    # shellcheck disable=SC2086
    kill -9 $remaining 2>/dev/null || true
  fi

  rm -f "$RUN_DIR/$name.pid"
  ok "$name stopped"
}

port_open() { (echo >"/dev/tcp/127.0.0.1/$1") >/dev/null 2>&1; }

confirm() {
  [[ $ASSUME_YES -eq 1 ]] && return 0
  read -r -p "  → $1 [Y/n] " reply </dev/tty || return 1
  [[ -z "$reply" || "$reply" =~ ^[Yy] ]]
}

pkg_install() {  # pkg_install <human name> <apt pkg> <dnf pkg> <brew pkg>
  local label="$2" ; local apt="$2" dnf="$3" brew="$4"
  if command -v apt-get >/dev/null 2>&1;  then confirm "Install $1 with apt?"  && sudo apt-get update -qq && sudo apt-get install -y "$apt"
  elif command -v dnf >/dev/null 2>&1;    then confirm "Install $1 with dnf?"  && sudo dnf install -y "$dnf"
  elif command -v pacman >/dev/null 2>&1; then confirm "Install $1 with pacman?" && sudo pacman -S --noconfirm "$dnf"
  elif command -v brew >/dev/null 2>&1;   then confirm "Install $1 with brew?" && brew install "$brew"
  else return 1; fi
}

# ─── modes that exit early ───────────────────────────────────────────────────

show_status() {
  step "Status"
  for n in api worker web; do
    if is_running "$n"; then ok "$n running (pid $(pid_of "$n"))"; else info "$n not running"; fi
  done
  if redis-cli ping >/dev/null 2>&1; then ok "redis responding"; else info "redis not responding"; fi
  if port_open 8000; then
    printf "\n"
    curl -s --max-time 5 http://localhost:8000/api/health 2>/dev/null | sed 's/^/  /' || true
    printf "\n"
  fi
}

case "$MODE" in
  stop)
    step "Stopping"
    for n in web worker api; do stop_one "$n"; done
    info "Redis left running — it is a system service, not ours to stop."
    exit 0 ;;
  logs)
    ls "$RUN_DIR"/*.log >/dev/null 2>&1 || die "No logs yet. Run ./scripts/start.sh first."
    exec tail -n 40 -f "$RUN_DIR"/*.log ;;
  status)
    show_status; exit 0 ;;
  restart)
    step "Stopping"
    for n in web worker api; do stop_one "$n"; done ;;
esac

# ═════════════════════════════════════════════════════════════════════════════
printf "${C_BOLD}${C_BLUE}Deep Truth${C_RESET} ${C_DIM}· %s${C_RESET}\n" "$REPO_ROOT"
MISSING=0

step "1. Toolchain"

PY=""
for c in python3 python; do
  command -v "$c" >/dev/null 2>&1 || continue
  if "$c" -c 'import sys; sys.exit(0 if sys.version_info>=(3,10) else 1)' 2>/dev/null; then PY="$c"; break; fi
done
if [[ -n "$PY" ]]; then
  ok "python $($PY -c 'import platform;print(platform.python_version())') ($PY)"
else
  bad "python 3.10+ not found"
  pkg_install "Python 3" python3 python3 python@3.12 || warn "Install Python 3.10+ manually"
  PY="python3"; MISSING=1
fi

if command -v node >/dev/null 2>&1; then
  ok "node $(node --version)"
else
  if [[ $WITH_WEB -eq 1 ]]; then
    bad "node not found (needed for the console; --no-web skips it)"
    pkg_install "Node.js" nodejs nodejs node || warn "Install Node 18+ manually"
    MISSING=1
  else
    info "node not found — skipped, --no-web given"
  fi
fi

if command -v ffmpeg >/dev/null 2>&1; then
  ok "ffmpeg present"
else
  warn "ffmpeg not found — PyAV usually bundles its own, so video may still work"
  pkg_install "ffmpeg" ffmpeg ffmpeg ffmpeg || true
fi

# ─── 2. Redis ────────────────────────────────────────────────────────────────
step "2. Redis"

if redis-cli ping >/dev/null 2>&1; then
  ok "redis responding on 6379"
else
  if command -v redis-server >/dev/null 2>&1; then
    warn "redis installed but not running"
    if [[ "$MODE" != "check" ]]; then
      if command -v systemctl >/dev/null 2>&1; then
        confirm "Start redis-server?" && sudo systemctl start redis-server 2>/dev/null \
          || sudo systemctl start redis 2>/dev/null || true
      elif command -v brew >/dev/null 2>&1; then
        brew services start redis || true
      fi
      sleep 1
      redis-cli ping >/dev/null 2>&1 && ok "redis started" || { bad "could not start redis"; MISSING=1; }
    fi
  else
    bad "redis not installed — the queue and cache cannot run without it"
    if [[ "$MODE" != "check" ]]; then
      pkg_install "Redis" redis-server redis redis || warn "Install Redis manually"
      command -v systemctl >/dev/null 2>&1 && sudo systemctl start redis-server 2>/dev/null || true
      redis-cli ping >/dev/null 2>&1 && ok "redis running" || MISSING=1
    else
      MISSING=1
    fi
  fi
fi

# ─── 3. Package import path ──────────────────────────────────────────────────
step "3. Package layout"

# `import deeptruth_pipeline` resolves to the repo root itself, so the
# directory has to carry that name. Renaming a cloned folder is intrusive, so a
# sibling symlink is used instead — it costs nothing and leaves the clone alone.
PARENT="$(dirname "$REPO_ROOT")"
LINK="$PARENT/deeptruth_pipeline"
if [[ "$(basename "$REPO_ROOT")" == "deeptruth_pipeline" ]]; then
  ok "repo directory is already named deeptruth_pipeline"
elif [[ -e "$LINK" ]]; then
  ok "import link present ($LINK)"
elif [[ "$MODE" == "check" ]]; then
  bad "missing import link: $LINK"; MISSING=1
else
  ln -s "$REPO_ROOT" "$LINK" && ok "created import link $LINK"
fi

# ─── 4. Python dependencies ──────────────────────────────────────────────────
step "4. Python dependencies"

MISSING_PY="$("$PY" - <<'PYEOF' 2>/dev/null || true
import importlib
need = {"torch":"torch","transformers":"transformers","peft":"peft","fastapi":"fastapi",
        "celery":"celery","redis":"redis","httpx":"httpx","reportlab":"reportlab",
        "aiosqlite":"aiosqlite","PIL":"pillow","cv2":"opencv-python","av":"av",
        "uvicorn":"uvicorn","reportlab.pdfgen":"reportlab"}
missing = []
for mod, pkg in need.items():
    try: importlib.import_module(mod)
    except Exception: missing.append(pkg)
print(" ".join(sorted(set(missing))))
PYEOF
)"

if [[ -z "${MISSING_PY// }" ]]; then
  ok "all Python packages present"
elif [[ "$MODE" == "check" ]]; then
  bad "missing: $MISSING_PY"; MISSING=1
else
  warn "missing: $MISSING_PY"
  info "installing from server/requirements.txt (this can take a few minutes)"
  "$PY" -m pip install -q -r "$SERVER_DIR/requirements.txt" || die "pip install failed"
  ok "Python dependencies installed"
fi

# ─── 5. Node dependencies ────────────────────────────────────────────────────
if [[ $WITH_WEB -eq 1 ]]; then
  step "5. Console dependencies"
  if [[ -d "$CLIENT_DIR/node_modules" ]]; then
    ok "node_modules present"
  elif [[ "$MODE" == "check" ]]; then
    bad "client/node_modules missing"; MISSING=1
  else
    info "running npm install (a few minutes on a cold cache)"
    ( cd "$CLIENT_DIR" && npm install --no-fund --no-audit >"$RUN_DIR/npm-install.log" 2>&1 ) \
      || die "npm install failed — see .run/npm-install.log"
    ok "console dependencies installed"
  fi
fi

# ─── 6. Model weights ────────────────────────────────────────────────────────
step "6. Model weights"

count_adapters() { find "$1" -maxdepth 2 -name adapter_config.json 2>/dev/null | wc -l | tr -d ' '; }
V="$(count_adapters "$REPO_ROOT/videos_checkpoints")"
I="$(count_adapters "$REPO_ROOT/images_checkpoints")"
[[ "$V" -gt 0 ]] && ok "video: $V adapters"  || { bad "no video adapters in videos_checkpoints/"; MISSING=1; }
[[ "$I" -gt 0 ]] && ok "image: $I adapters"  || { bad "no image adapters in images_checkpoints/"; MISSING=1; }

ls "$REPO_ROOT"/checkpoints/audios_checkpoints/*.pt >/dev/null 2>&1 \
  && ok "audio checkpoint present" \
  || info "audio: no .pt — engine stays a stub (expected; weights are gitignored)"
ls "$REPO_ROOT"/checkpoints/srm_checkpoints/*.pt >/dev/null 2>&1 \
  && ok "SRM checkpoint present" \
  || info "SRM: no .pt — features only, no verdict (expected)"

if [[ "$MODE" == "check" ]]; then
  printf "\n"
  [[ $MISSING -eq 0 ]] && printf "${C_GREEN}Everything needed is present.${C_RESET}\n" \
                       || printf "${C_YELLOW}Some things are missing (see ✗ above). Run without --check to install.${C_RESET}\n"
  exit $MISSING
fi

[[ $MISSING -eq 0 ]] || die "Cannot start — unresolved problems above."

# ─── 7. Start ────────────────────────────────────────────────────────────────
step "7. Starting services"

if port_open 8000 && ! is_running api; then
  warn "port 8000 is already in use by something this script did not start"
else
  start_bg api "$SERVER_DIR" "$PY" main.py
fi

# Workers do not reload on code change — that is why --restart exists.
start_bg worker "$SERVER_DIR" \
  celery -A app.queue.celery_app worker \
  --loglevel=info --concurrency="$WORKERS" --queues=analysis \
  --hostname="w1@%h" --without-gossip --without-mingle

if [[ $WITH_WEB -eq 1 ]]; then
  if port_open 3000 && ! is_running web; then
    warn "port 3000 already in use by something this script did not start"
  else
    start_bg web "$CLIENT_DIR" npm run dev
  fi
fi

# ─── 8. Health ───────────────────────────────────────────────────────────────
step "8. Health"

printf "  waiting for the API"
for _ in $(seq 1 45); do
  curl -s --max-time 2 http://localhost:8000/api/health >/dev/null 2>&1 && break
  printf "."; sleep 1
done
printf "\n"

HEALTH="$(curl -s --max-time 5 http://localhost:8000/api/health 2>/dev/null || echo '')"
if [[ -z "$HEALTH" ]]; then
  bad "API not answering — see .run/api.log"
else
  ok "API: $HEALTH"
  # "idle" means Redis is fine but no worker answered a control ping. Worth
  # calling out: uploads queue and never run, which looks like a hang.
  grep -q '"status":"ok"' <<<"$HEALTH" && ok "workers connected" \
    || warn "no worker registered yet — give it a few seconds, then ./scripts/start.sh --status"
fi

printf "\n${C_BOLD}${C_GREEN}Running.${C_RESET}\n"
printf "  Console   ${C_BLUE}http://localhost:3000${C_RESET}\n"
printf "  API docs  ${C_BLUE}http://localhost:8000/api/docs${C_RESET}\n"
printf "\n  ${C_DIM}logs${C_RESET}    ./scripts/start.sh --logs\n"
printf "  ${C_DIM}status${C_RESET}  ./scripts/start.sh --status\n"
printf "  ${C_DIM}stop${C_RESET}    ./scripts/start.sh --stop\n"
printf "\n  ${C_YELLOW}Note${C_RESET} the worker does not hot-reload. After changing Python code:\n"
printf "       ./scripts/start.sh --restart\n\n"
