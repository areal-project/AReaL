#!/usr/bin/env bash
set -euo pipefail

mode=$1
state=$2
branch=ascend-nightly-results
remote=$(git remote get-url origin)
git init -q -b "$branch" "$state"
git -C "$state" remote add origin "$remote"
ref=$(git ls-remote --heads "$remote" "refs/heads/$branch")
if [[ -n "$ref" ]]; then
    git -C "$state" fetch -q origin "refs/heads/$branch"
    git -C "$state" merge --ff-only FETCH_HEAD
fi
git -C "$state" config user.name 'github-actions[bot]'
git -C "$state" config user.email '41898282+github-actions[bot]@users.noreply.github.com'

case "$mode" in
    prepare)
        sha=$(git rev-parse HEAD)
        if [[ -f "$state/last-attempt" ]]; then
            previous=$(cat "$state/last-attempt")
            if git cat-file -e "$previous^{commit}" 2>/dev/null; then
                if git diff --quiet "$previous" "$sha" -- . ':!docs/**' ':!*.md' ':!*.rst'; then
                    echo "Skipped: no code changes since the previous attempt." >> "$GITHUB_STEP_SUMMARY"
                    exit 0
                fi
            fi
        fi
        echo "$sha" > "$state/last-attempt"
        git -C "$state" add last-attempt
        git -C "$state" commit -qm "Record nightly attempt for $sha"
        git -C "$state" push origin "HEAD:refs/heads/$branch"
        echo "sha=$sha" >> "$GITHUB_OUTPUT"
        echo "baseline=$(git -C "$state" rev-parse HEAD)" >> "$GITHUB_OUTPUT"
        ;;
    publish)
        for suite in grpo sft; do
            snapshot="artifacts/ascend-nightly-$suite/metrics.json"
            if [[ ! -f "$snapshot" ]]; then
                echo "No verified metrics available for ${suite^^}." >> "$GITHUB_STEP_SUMMARY"
                echo >> "$GITHUB_STEP_SUMMARY"
                continue
            fi
            args=()
            if git -C "$state" cat-file -e "$BASELINE:$suite/metrics.json" 2>/dev/null; then
                git -C "$state" show "$BASELINE:$suite/metrics.json" > "$RUNNER_TEMP/$suite-baseline.json"
                args+=(--baseline-snapshot "$RUNNER_TEMP/$suite-baseline.json")
            fi
            python scripts/ascend_nightly_grpo_report.py --suite "$suite" --snapshot "$snapshot" \
                --output "nightly-reports/$suite.md" --summary-output "$RUNNER_TEMP/$suite-summary.md" "${args[@]}"
            cat "$RUNNER_TEMP/$suite-summary.md" >> "$GITHUB_STEP_SUMMARY"
            echo >> "$GITHUB_STEP_SUMMARY"
            mkdir -p "$state/$suite"
            cp "$snapshot" "$state/$suite/metrics.json"
            git -C "$state" add "$suite/metrics.json"
        done
        if [[ $(cat "$state/last-attempt") == "$(git rev-parse HEAD)" ]] && ! git -C "$state" diff --cached --quiet; then
            git -C "$state" commit -qm "Save nightly metrics for $(git rev-parse --short HEAD)"
            git -C "$state" push origin "HEAD:refs/heads/$branch"
        fi
        ;;
    *) exit 2 ;;
esac
