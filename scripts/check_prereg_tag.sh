#!/usr/bin/env bash
# Phase 1 standing check: the current freeze tag has not been rewritten, and the files it
# froze have not been edited since. A silently edited frozen config is the one failure mode
# that invalidates every result downstream of it, so this is a check with an exit code rather
# than a line in a checklist.
#
# Tag history: prereg-v1 (initial freeze) -> prereg-v1.1 (re-frozen after the Phase 0
# GPU-verification architecture record was added to config/frozen_thresholds.yaml; see
# preregistration/DEVIATIONS.md, "Architecture record is preliminary"). Bump TAG here, with a
# DEVIATIONS.md entry and a new annotated tag, on the next genuine deviation — never edit a
# frozen file under an existing tag.
set -uo pipefail

TAG=${TAG:-prereg-v1.1}
FROZEN_PATHS=${FROZEN_PATHS:-"config/analysis_config.yaml config/frozen_thresholds.yaml preregistration/PREREGISTRATION.md splits/splits_final.json splits/fold_assignment.csv"}
status=0

if ! git rev-parse -q --verify "refs/tags/$TAG" >/dev/null; then
  echo "FAIL: tag '$TAG' does not exist. Nothing is frozen yet."
  echo "      Freeze with: git tag -a $TAG -m 'pre-registration frozen' && git push origin $TAG"
  exit 1
fi

TAG_COMMIT=$(git rev-list -n 1 "$TAG")
echo "Tag $TAG -> $TAG_COMMIT"

# The tag must still be an ancestor of HEAD; if it is not, history was rewritten under it.
if git merge-base --is-ancestor "$TAG_COMMIT" HEAD; then
  echo "OK:   $TAG is an ancestor of HEAD (no history rewrite)."
else
  echo "FAIL: $TAG is NOT an ancestor of HEAD — history was rewritten or the branch diverged."
  status=1
fi

if git ls-remote --exit-code --tags origin "refs/tags/$TAG" >/dev/null 2>&1; then
  REMOTE_COMMIT=$(git ls-remote --tags origin "refs/tags/$TAG" | awk '{print $1}' | head -1)
  REMOTE_TARGET=$(git rev-list -n 1 "$REMOTE_COMMIT" 2>/dev/null || echo "$REMOTE_COMMIT")
  if [ "$REMOTE_TARGET" = "$TAG_COMMIT" ]; then
    echo "OK:   origin's $TAG points at the same commit."
  else
    echo "FAIL: origin's $TAG points at $REMOTE_TARGET, local at $TAG_COMMIT — the tag moved."
    status=1
  fi
else
  echo "WARN: $TAG is not on origin. A pre-registration nobody else can see is not one."
fi

echo
echo "Frozen files, tag -> HEAD:"
for p in $FROZEN_PATHS; do
  if ! git cat-file -e "$TAG_COMMIT:$p" 2>/dev/null; then
    echo "  WARN  $p did not exist at $TAG"
    continue
  fi
  if git diff --quiet "$TAG_COMMIT" HEAD -- "$p"; then
    echo "  OK    $p unchanged"
  else
    echo "  FAIL  $p CHANGED since $TAG:"
    git diff --stat "$TAG_COMMIT" HEAD -- "$p" | sed 's/^/          /'
    echo "          If this change is genuine it is a DEVIATION: record it in"
    echo "          preregistration/DEVIATIONS.md and cut a new tag. Never a silent edit."
    status=1
  fi
done

echo
[ $status -eq 0 ] && echo "PASS: pre-registration intact." || echo "FAIL: see above."
exit $status
