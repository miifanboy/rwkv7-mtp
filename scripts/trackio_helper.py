"""Shared, restart-safe trackio integration for scripts/train_mtp.py and
scripts/generate_data.py.

Restart safety (train_mtp.py)
-----------------------------
Every metric record is first buffered in memory and only forwarded to trackio
by ``flush()``, which the trainer calls right AFTER a checkpoint has been
successfully written.  Consequences:

* A run killed between checkpoint saves leaves the trackio log ending exactly
  at the last saved step (the steps after it were buffered and are lost -- the
  same steps are equally absent from the checkpoint, so log and weights stay
  consistent).
* On restart, ``init(..., resume="allow")`` with the same run name continues
  the SAME trackio run (same run_id), and records are logged with explicit
  ``step=`` values, so the dashboard picks up exactly where the checkpoint is.
  No orphaned rows beyond the checkpoint are ever created, so there is nothing
  to delete afterwards -- which matters because trackio has no remote
  "delete rows with step > N" API for Space-backed projects.

"step" is a trackio RESERVED_KEY: leaving it inside the payload would get it
silently renamed to "__step", so ``flush()`` pops it and passes it as the
explicit ``step=`` argument of ``trackio.log()``.

Everything degrades gracefully: if trackio is not installed or ``init`` fails,
tracking disables itself with a warning and the training/generation loops run
untracked.  Trackio itself is also written defensively (its internal failures
degrade to warnings), so logging never takes down a training run.
"""

from __future__ import annotations


class TrackioRun:
    """Buffered trackio run wrapper; a no-op when disabled.

    Logs to the LOCAL trackio database only (no HF Space): remote Space
    deployment is intentionally not wired up -- see the trackio section in
    MTP_README.md for how to publish a local project to a free static Space
    with trackio.sync(sdk='static') if a remote dashboard is ever wanted.
    """

    def __init__(self, project: str, name: str | None = None,
                 config: dict | None = None, tag: str = "trackio"):
        self.enabled = False
        self.run = None
        self._trackio = None
        self._buf: list[dict] = []
        self._tag = tag
        if not project:
            return
        try:
            import trackio
        except ImportError as e:
            print(f"[{tag}] trackio is not installed ({e}); tracking disabled "
                  f"(pip install trackio)")
            return
        try:
            self.run = trackio.init(project=project, name=name, config=config,
                                    resume="allow")
            self._trackio = trackio
            self.enabled = True
            url = getattr(self.run, "url", None)
            print(f"[{tag}] tracking run '{name or self.run.name}' in local "
                  f"project '{project}'"
                  + (f": {url}" if url else "")
                  + " (view: trackio show --project " + project + ")")
        except Exception as e:
            print(f"[{tag}] init failed ({e}); tracking disabled")

    def buffer(self, record: dict) -> None:
        """Queue one metric record for the next flush().  Copies the record so
        the caller may keep mutating its dict afterwards."""
        if self.enabled:
            self._buf.append(dict(record))

    def flush(self) -> None:
        """Log every buffered record, then clear the buffer.

        train_mtp.py calls this right after a checkpoint has been written, so
        the flushed step range always corresponds to a real checkpoint.  A
        trackio failure here must never fail the training run.
        """
        if not self.enabled or not self._buf:
            return
        buf, self._buf = self._buf, []
        try:
            for rec in buf:
                step = rec.pop("step", None)
                self._trackio.log(rec, step=step)
        except Exception as e:
            print(f"[{self._tag}] WARNING: flush failed ({e}); "
                  f"{len(buf)} buffered records lost")

    def log_now(self, metrics: dict, step: int | None = None) -> None:
        """Log immediately (used by generate_data.py, which has no checkpoints
        to gate the flush on)."""
        if not self.enabled:
            return
        try:
            self._trackio.log(metrics, step=step)
        except Exception as e:
            print(f"[{self._tag}] WARNING: log failed ({e})")

    def finish(self) -> None:
        """Flush whatever is buffered and mark the run completed."""
        self.flush()
        if not self.enabled:
            return
        try:
            self._trackio.finish()
            self.enabled = False
        except Exception as e:
            print(f"[{self._tag}] WARNING: finish failed ({e})")
