#!/usr/bin/env bash
# ops/watchdog.sh — keeps the Saudi TTS agent thinking, never stalled.
# Run it inside the tmux session "watchdog":   bash ~/saudi-tts/ops/watchdog.sh
# It never touches training jobs; it only (re)starts and nudges the Claude Code
# session that lives in the tmux session "agent".
set -u
cd "$HOME/saudi-tts" || exit 1
mkdir -p ops

LOG=ops/watchdog.log
MODE_FILE=ops/AUTH_MODE                          # "login" (full login, Remote Control works) | "token" (setup-token) | "api" (fallback key)
FALLBACK="$HOME/.secrets/fallback.env"           # optional: export ANTHROPIC_API_KEY=sk-ant-api... (rung 4 only)
CHECK_SEC=300                                    # look every 5 minutes
STALE_SEC=5400                                   # nudge if PROGRESS.md is quiet for 90 minutes
NUDGE_GAP_SEC=5400                               # at most one nudge per 90 minutes
API_RETRY_SEC=43200                              # in API mode, try the subscription login again after 12 h

GOAL="/goal Act as the autonomous Saudi TTS researcher described in docs/plan-v2.md and docs/kickoff.md (kickoff wins where they differ; start with the bootstrap stage, inventorying the S3 bucket and the TTS-FT server read-only first). Done means: PLAN.md shows every stage PASS with gate numbers in PROGRESS.md, MODEL_CARD.md and the latency report are in reports/, and the final checkpoint, voice library and reports are synced to s3://aljawab-asr-data/saudi-tts/ and verified before Sun 1 Nov 09:00 London."
NUDGE="Watchdog: resume work. Read PLAN.md and the last 40 lines of PROGRESS.md, then continue the first unfinished stage. Append a heartbeat line to PROGRESS.md at least every 60 minutes."

case "$(cat "$MODE_FILE" 2>/dev/null)" in login|token|api) ;; *) echo login > "$MODE_FILE" ;; esac   # default and migrate old values
last_nudge=0
api_since=0

say()   { tmux send-keys -t agent -l "$1"; sleep 1; tmux send-keys -t agent Enter; }   # type text literally, then Enter
log()   { echo "$(date -Is) $*" | tee -a "$LOG"; }
mode()  { cat "$MODE_FILE"; }
pane()  { tmux capture-pane -p -t agent -S -40 2>/dev/null; }
# Running = the agent pane's foreground process is not a bare shell (robust to how the claude binary names itself).
agent_running() {
  tmux has-session -t agent 2>/dev/null || return 1
  case "$(tmux display -p -t agent '#{pane_current_command}')" in
    bash|sh|zsh|dash|"") return 1 ;;
    *) return 0 ;;
  esac
}

start_agent() {
  tmux has-session -t agent 2>/dev/null || tmux new -d -s agent -c "$HOME/saudi-tts"
  local auth
  case "$(mode)" in
    login) auth='unset CLAUDE_CODE_OAUTH_TOKEN ANTHROPIC_API_KEY' ;;
    token) auth='unset ANTHROPIC_API_KEY; source ~/.secrets/claude.env' ;;
    api)   auth='source ~/.secrets/claude.env; source ~/.secrets/fallback.env' ;;
  esac
  log "starting agent session (auth: $(mode))"
  say "clear"; tmux clear-history -t agent   # so old messages in the pane are not mistaken for new ones
  tmux send-keys -t agent "bash -c '$auth; export CLAUDE_CODE_RETRY_WATCHDOG=1; exec claude --continue --permission-mode auto'" Enter
  sleep 90
  [ "$(mode)" = login ] && { say "/remote-control"; sleep 10; }
  say "$GOAL"
  last_nudge=$(date +%s)
}

# Next auth mode after a failure in the current one.
fallback_mode() {
  case "$(mode)" in
    login) echo token ;;
    token) [ -f "$FALLBACK" ] && echo api || echo token ;;
    api)   echo token ;;
  esac
}

log "watchdog up (mode: $(mode))"
while [ ! -f DONE ]; do
  now=$(date +%s)

  if ! agent_running; then
    start_agent
  else
    screen="$(pane)"

    # Login expired or rejected -> fall back to the one-year token (then to the API key).
    if echo "$screen" | grep -qiE "run /login|claude auth login|login (has )?expired|token (has )?expired|authentication_error|invalid api key|not logged in"; then
      next=$(fallback_mode)
      if [ "$next" != "$(mode)" ]; then
        log "auth problem in $(mode) mode -> switching to $next"
        echo "$next" > "$MODE_FILE"; [ "$next" = api ] && api_since=$now
        say "/exit"; sleep 15
        continue
      fi
    fi

    # Rung 4: weekly limit on the subscription -> switch to the API key, if one is configured.
    if echo "$screen" | grep -qiE "hit your weekly limit"; then
      if [ "$(mode)" != api ] && [ -f "$FALLBACK" ]; then
        log "weekly limit seen -> switching agent to API key"
        echo api > "$MODE_FILE"; api_since=$now
        say "/exit"; sleep 15
        continue
      elif [ ! -f "$FALLBACK" ]; then
        log "weekly limit seen and no fallback key configured; training continues, agent waits"
      fi
    fi

    # Model-specific weekly cap -> move to the other model and keep going.
    if echo "$screen" | grep -qiE "hit your Opus limit"; then
      log "Opus limit seen -> switching to Sonnet"
      say "/model sonnet"; sleep 5
      say "$NUDGE"; last_nudge=$now
    fi

    # Back to the subscription after 12 h on the API key.
    if [ "$(mode)" = api ] && [ $((now - api_since)) -gt $API_RETRY_SEC ]; then
      log "12 h on API key -> trying the subscription login again"
      echo login > "$MODE_FILE"
      say "/exit"; sleep 15
      continue
    fi

    # Alive but quiet (goal cleared, stuck, waiting): nudge, unless it is inside a short limit wait.
    if [ -f PROGRESS.md ]; then
      age=$((now - $(stat -c %Y PROGRESS.md)))
      if [ $age -gt $STALE_SEC ] && [ $((now - last_nudge)) -gt $NUDGE_GAP_SEC ] \
         && ! echo "$screen" | grep -qiE "limit.*reset|resets [0-9]"; then
        log "PROGRESS.md quiet for $((age / 60)) min -> nudging"
        say "$GOAL"; sleep 5
        say "$NUDGE"
        last_nudge=$now
      fi
    fi
  fi

  sleep "$CHECK_SEC"
done
log "DONE file found; watchdog exiting"
