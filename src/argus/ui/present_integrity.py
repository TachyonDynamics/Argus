"""View model for the Integrity page. Pure logic, no web code, so it's easy to test.

Only metadata is ever shown: model names, digest prefixes, sizes, timestamps and
file paths. Model names and paths come from the disk, so they are treated as
untrusted: clipped in length here, HTML-escaped by the template, and only ever
placed inside a copy-and-paste shell command if they look completely ordinary.
"""
from __future__ import annotations

import re

from argus.format import DASH, fmt_ago, fmt_size
from argus.integrity.models import (BLOB_CORRUPT, BLOB_OK, BLOB_UNREADABLE, MANIFEST_APPROVED,
                                    MANIFEST_CHANGED, MANIFEST_MISSING, MANIFEST_NEW,
                                    MANIFEST_UNREADABLE, BlobFinding, ManifestFinding,
                                    ScanResult)
from argus.integrity.service import RECHECK_DAYS

_MAX_NAME = 120
# What a normal Ollama model reference looks like. Anything else is never pasted
# into a suggested shell command (a folder can be named to look like a command).
_PLAIN_NAME = re.compile(r"^[A-Za-z0-9._/-]{1,80}:[A-Za-z0-9._-]{1,40}$")

INTRO = ("Argus checks that your Ollama model files still match what you last approved, "
        "so you'd know if one was corrupted or swapped without your knowledge. This is "
        "informational only - it never stops Ollama from running a model.")

NEW_HELP = ("These models haven't been reviewed yet. Approving one tells Argus what it "
           "should look like going forward, so you'll see a warning here if it ever "
           "changes without an ollama pull or ollama create you ran.")
MISSING_HELP = ("These were approved before but aren't installed anymore (for example "
                "after ollama rm). Removing them here just tidies up Argus's own list - "
                "it doesn't touch Ollama or any files.")
UNREADABLE_HELP = ("Argus couldn't open these files to check them, often because another "
                   "program had them locked at the time. On its own this isn't a sign of "
                   "a problem - try checking again.")

MESSAGES = {
    "scan_started": "Checking your model files now. This page will update when it finishes.",
    "busy": "A check is already running. Try again when it finishes.",
    "approved": "Approved. Argus will let you know if this model ever changes.",
    "approve_failed": ("Couldn't approve that: the model changed since this page was "
                       "loaded, or a check is running. Check again and review it."),
    "nothing_to_approve": "There were no unreviewed models to approve.",
    "forgotten": "Removed from the list.",
    "forget_failed": "Couldn't remove that one.",
}

_STATUS = {
    MANIFEST_APPROVED: ("Approved", "ok"),
    MANIFEST_NEW: ("Not approved yet", "neutral"),
    MANIFEST_CHANGED: ("Changed since approved", "bad"),
    MANIFEST_MISSING: ("Removed", "neutral"),
    MANIFEST_UNREADABLE: ("Couldn't read", "neutral"),
}


def short(digest: str | None) -> str:
    return digest[:12] if digest else DASH


def clip(text: str, limit: int = _MAX_NAME) -> str:
    return text if len(text) <= limit else text[: limit - 1] + "…"


def _reinstall_command(names: list[str]) -> str | None:
    """`ollama rm X` then `ollama pull X`, but only when there is exactly one
    model and its name is plainly safe to paste into a shell."""
    if len(names) == 1 and _PLAIN_NAME.match(names[0]):
        n = names[0]
        return f"ollama rm {n}   then   ollama pull {n}"
    return None


def _corrupt_blob(b: BlobFinding) -> dict[str, object]:
    who = [clip(m) for m in b.models]
    if b.models:
        cmd = _reinstall_command(b.models)
        fix = ("Remove the model and download it again"
               + (f": {cmd}" if cmd else " with ollama rm and ollama pull") + ".")
    else:
        fix = ("No installed model refers to this file, so it may be left over from an "
               "earlier download. If you don't recognize it, look into where it came "
               "from before deleting it.")
    return {
        "kind": "blob_corrupt",
        "kicker": "Confirmed",
        "title": "A model file doesn't match its fingerprint",
        "subject": ", ".join(who) if who else "(no installed model uses this file)",
        "expected": short(b.digest),
        "found": short(b.actual_digest),
        "size": fmt_size(b.size),
        "path": clip(str(b.path), 300),
        "explanation": ("The file's contents no longer match the fingerprint it was saved "
                        "under, so its bytes changed after it was written. Disk errors or an "
                        "interrupted copy are the more common causes, but deliberate "
                        "modification is possible; Argus can't tell which."),
        "fix": fix,
    }


def _changed_manifest(m: ManifestFinding, now_wall_ns: int) -> dict[str, object]:
    return {
        "kind": "manifest_changed",
        "kicker": "Needs your review",
        "title": f"{clip(m.name)} changed since you approved it",
        "name": m.name,
        "digest": m.current_digest or "",
        "approved": short(m.approved_digest),
        "found": short(m.current_digest),
        "approved_when": _ago(m.approved_wall_ns, now_wall_ns),
        "explanation": ("This model's file listing is different from the one you approved. "
                        "If you ran ollama pull or ollama create for this model recently, "
                        "that's expected - approve the new version below. If you didn't, "
                        "hold off approving until you know why it changed."),
    }


def _ago(wall_ns: int | None, now_wall_ns: int) -> str:
    return DASH if wall_ns is None else fmt_ago((now_wall_ns - wall_ns) / 1e9)


def _model_row(m: ManifestFinding, blobs: dict[str, BlobFinding], now_wall_ns: int) -> dict[str, object]:
    label, tone = _STATUS.get(m.status, (m.status, "neutral"))
    files = ""
    if m.blob_digests:
        present = [blobs[d] for d in m.blob_digests if d in blobs]
        not_on_disk = len(m.blob_digests) - len(present)
        bad = sum(1 for b in present if b.status == BLOB_CORRUPT)
        unread = sum(1 for b in present if b.status == BLOB_UNREADABLE)
        ok = sum(1 for b in present if b.status == BLOB_OK)
        parts = [f"{bad} of {len(m.blob_digests)} files failed" if bad
                 else f"{ok} of {len(m.blob_digests)} files verified"]
        if unread:
            parts.append(f"{unread} couldn't be read")
        if not_on_disk:
            parts.append(f"{not_on_disk} not on disk")
        files = ", ".join(parts)
    approved = ("approved " + _ago(m.approved_wall_ns, now_wall_ns)
                if m.status == MANIFEST_APPROVED else "")
    return {"name": clip(m.name), "status": m.status, "status_label": label, "tone": tone,
            "files": files, "approved": approved}


def _coverage(result: ScanResult) -> str:
    checked = sum(1 for b in result.blobs if b.rehashed and b.status != BLOB_UNREADABLE)
    cached = sum(1 for b in result.blobs if not b.rehashed)
    if not result.blobs:
        return ""
    if result.forced:
        return f"Every file was checked just now ({checked})."
    if cached == 0:
        return f"All {checked} file{'s' if checked != 1 else ''} were checked just now."
    return (f"{checked} file{'s' if checked != 1 else ''} checked just now; {cached} already "
            f"verified within the last {RECHECK_DAYS} days and skipped this time. "
            "Use \"Check everything again\" to check every file regardless.")


def present_integrity(result: ScanResult | None, *, scanning: bool, now_wall_ns: int,
                      msg: str | None = None, msg_count: int | None = None) -> dict[str, object]:
    notice = MESSAGES.get(msg or "")
    if msg == "approved_all" and msg_count is not None:
        notice = f"Approved {msg_count} model{'s' if msg_count != 1 else ''}."
    view: dict[str, object] = {
        "intro": INTRO, "new_help": NEW_HELP, "missing_help": MISSING_HELP,
        "unreadable_help": UNREADABLE_HELP,
        "notice": notice, "scanning": scanning, "can_edit": not scanning,
        "warnings": [], "review_new": [], "review_unreadable": [], "review_missing": [],
        "models": [], "coverage": "", "last_checked": "", "state": "unscanned",
        "headline": "", "subline": "",
    }
    if result is None:
        view["headline"] = ("Checking model files…" if scanning
                            else "Model files haven't been checked yet")
        view["subline"] = ("" if scanning else
                           "The first check starts automatically when Argus launches. "
                           "You can also start one now.")
        return view

    view["last_checked"] = _ago(result.scanned_wall_ns, now_wall_ns)
    if result.models_dir is None:
        view.update(state="no_models", headline="No Ollama model folder found",
                    subline=("Argus looks in the folder named by the OLLAMA_MODELS environment "
                             "variable, or else in the .ollama\\models folder inside your user "
                             "profile. Neither exists on this computer - this usually just "
                             "means Ollama hasn't been run yet."))
        return view
    if not result.blobs and not result.manifests:
        view.update(state="empty", headline="No models installed yet",
                    subline="The model folder exists but has no models in it.")
        return view

    blobs = {b.digest: b for b in result.blobs}
    view["warnings"] = ([_corrupt_blob(b) for b in result.corrupt_blobs]
                        + [_changed_manifest(m, now_wall_ns) for m in result.changed_manifests])
    view["review_new"] = [{"name": clip(m.name), "raw_name": m.name, "digest": m.current_digest or ""}
                          for m in result.new_manifests]
    view["review_unreadable"] = [{"digest": short(b.digest), "models": ", ".join(clip(x) for x in b.models),
                                  "size": fmt_size(b.size), "path": clip(str(b.path), 300)}
                                 for b in result.unreadable_blobs]
    view["review_missing"] = [{"name": clip(m.name), "raw_name": m.name}
                              for m in result.missing_manifests]
    view["models"] = [_model_row(m, blobs, now_wall_ns) for m in result.manifests]
    view["coverage"] = _coverage(result)
    n = result.warning_count
    if n:
        view.update(state="warnings",
                    headline=f"{n} model file problem{'s' if n != 1 else ''} need{'s' if n == 1 else ''} your attention")
    else:
        view.update(state="clean", headline="All checked model files match their fingerprints")
    return view