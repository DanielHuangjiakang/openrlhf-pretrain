#!/usr/bin/env bash
#
# Drive cmux's native progress bar from the pipeline running on the rented GPU
# box, and show the full pipeline view underneath it.
#
#   bash scripts/cmux-progress.sh           # poll every 20s
#   INTERVAL=60 bash scripts/cmux-progress.sh
#   GPU_HOST=vast-2 bash scripts/cmux-progress.sh
#
# cmux has first-class progress and status APIs -- `cmux set-progress` and
# `cmux set-status` -- so the bar is real workspace chrome in the right sidebar,
# not text in a pane. It stays visible whichever pane is focused, which is the
# point: the pipeline runs for hours and the question is always "how far in".
#
# Targeting: cmux resolves the workspace from CMUX_WORKSPACE_ID in the
# environment, so this has to run in a cmux terminal. A Dock control is its
# intended home (see .cmux/dock.json); any cmux pane works. Run it outside cmux
# and set-progress has no workspace to update -- the text view still works.
#
# The remote side is scripts/progress.py, already on the box. One ssh round trip
# per tick fetches both the full render and the one-line summary; the ssh alias
# carries ControlMaster, so a tick costs ~0.5s rather than a fresh handshake.
set -uo pipefail

HOST="${GPU_HOST:-vast-4090}"
PROJ="${GPU_PROJ:-/root/openrlhf-pretrain}"
INTERVAL="${INTERVAL:-20}"
CMUX="${CMUX_BUNDLED_CLI_PATH:-$(command -v cmux 2>/dev/null)}"
[[ -x "${CMUX:-}" ]] || CMUX=/Applications/cmux.app/Contents/Resources/bin/cmux
[[ -x "$CMUX" ]] || { echo "找不到 cmux CLI，只显示文本视图" >&2; CMUX=""; }

cmuxq() { [[ -n "$CMUX" ]] && "$CMUX" "$@" >/dev/null 2>&1; return 0; }

# A bar frozen at 40% after the poller dies is worse than no bar at all.
cleanup() {
    cmuxq clear-progress
    cmuxq clear-status pipeline
    cmuxq clear-status gpu
    exit 0
}
trap cleanup INT TERM EXIT

# progress.py --oneline prints e.g.
#   tg30-2ep 141/232 61% 52s/it eta 1:18:45 gpu18% frac=0.6078
# or one of the wordy states 空闲 / 评测中 / 阶段切换 when no run is live. The
# trailing frac= is the contract: progress.py owns the arithmetic (episodes fold
# in there), this script owns the display.
REMOTE="cd '$PROJ' && PROJ='$PROJ' python3 scripts/progress.py"

while true; do
    # stderr is dropped on purpose: vast.ai prints a login banner there on every
    # new master connection, which would otherwise land in the pane.
    out=$(ssh -o BatchMode=yes -o ConnectTimeout=10 "$HOST" \
              "$REMOTE && printf '@@BAR@@%s\n' \"\$($REMOTE --oneline)\"" 2>/dev/null)
    rc=$?

    bar=$(printf '%s\n' "$out" | sed -n 's/^@@BAR@@//p' | tail -1)
    view=$(printf '%s\n' "$out" | grep -v '^@@BAR@@')

    if [[ $rc -ne 0 || -z "$out" ]]; then
        cmuxq clear-progress
        cmuxq set-status pipeline "机器不可达" --icon exclamationmark.triangle --color "#f85149"
        cmuxq clear-status gpu
        view="  ✗ 连不上 $HOST（机器关了，或网络断了）"
        bar=""
    fi

    gpu=""; frac=""
    [[ "$bar" =~ gpu([0-9]+)% ]] && gpu="${BASH_REMATCH[1]}%"
    [[ "$bar" =~ frac=([0-9.]+) ]] && frac="${BASH_REMATCH[1]}"
    # The bar gets the human-readable half; gpu and frac have their own homes.
    label=$(printf '%s' "$bar" | sed -e 's/ *frac=[0-9.]*//' -e 's/ *gpu[0-9]*%//')

    case "$bar" in
        "")
            : ;;                                    # unreachable, already reported
        *frac=*)
            cmuxq set-progress "$frac" --label "$label"
            cmuxq set-status pipeline "$label" --icon bolt.fill --color "#7ee787"
            ;;
        *评测中*|*阶段切换*)
            cmuxq clear-progress                     # no step count to derive a fraction from
            cmuxq set-status pipeline "$bar" --icon hourglass --color "#d29922"
            ;;
        *)
            cmuxq clear-progress
            cmuxq set-status pipeline "GPU 空闲" --icon moon --color "#8b949e"
            ;;
    esac

    if [[ -n "$gpu" ]]; then
        cmuxq set-status gpu "GPU $gpu" --icon thermometer --color "#58a6ff"
    else
        cmuxq clear-status gpu
    fi

    printf '\033[2J\033[H%s\n\n  %s  ·  %ss 后刷新  ·  Ctrl-C 停止（会清掉进度条）\n' \
           "$view" "$(date '+%H:%M:%S %Z')" "$INTERVAL"
    sleep "$INTERVAL"
done
