#!/usr/bin/env python3
"""
Sync training runs from HPC to local machine.

Handles Kerberos/pkinit authentication (smartcard + PIN) and retries
dropped rsync connections automatically. Only re-authenticates when
the Kerberos ticket expires, so you enter your PIN once and it stays
valid for the ticket lifetime (typically 10-24 hours).

Usage:
    python utils/sync_remote_runs.py                    # sync all runs from makau
    python utils/sync_remote_runs.py --remote coral     # sync all runs from coral
    python utils/sync_remote_runs.py --run ppo_optomech_1_1774138836  # sync one run
    python utils/sync_remote_runs.py --interval 60      # poll every 60s
    python utils/sync_remote_runs.py --dry-run           # show what would transfer
    python utils/sync_remote_runs.py --bootstrap        # sync full bootstrap_runs/
    python utils/sync_remote_runs.py --bootstrap --run bootstrap_1776285401
                                                        # sync one bootstrap run id
"""

import argparse
import subprocess
import sys
import time
from pathlib import Path

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
PKINIT_BIN = "/usr/local/krb5/bin/pkinit"
KLIST_BIN = "/usr/local/krb5/bin/klist"
KDESTROY_BIN = "/usr/local/krb5/bin/kdestroy"
PRINCIPAL = "fletch@HPCMP.HPC.MIL"

REMOTES = {
    "makau": {
        "host": "fletch@makau.mhpcc.hpc.mil",
        "runs": "/p/home/fletch/visuomotor-deep-optics/runs/",
        "bootstrap_runs": "/p/home/fletch/visuomotor-deep-optics/bootstrap_runs/",
    },
    "coral": {
        "host": "fletch@coral.mhpcc.hpc.mil",
        "runs": "/wdata/home/fletch/visuomotor-deep-optics/runs/",
        "bootstrap_runs": "/wdata/home/fletch/visuomotor-deep-optics/bootstrap_runs/",
    },
}
DEFAULT_REMOTE = "makau"
_REPO_ROOT = Path(__file__).resolve().parent.parent
LOCAL_RUNS = _REPO_ROOT / "runs"
LOCAL_BOOTSTRAP_RUNS = _REPO_ROOT / "bootstrap_runs"

# SSH: fresh connection per rsync invocation. We previously used
# ControlMaster multiplexing to avoid repeated auth, but stale/broken
# sockets in /tmp silently stalled rsync file-list exchanges for
# hours. Kerberos ticket caching already makes reconnects fast (no
# smartcard re-prompt while the ticket is valid), so the only cost of
# dropping multiplexing is a ~1 s TCP + shell-init overhead per call.
SSH_CONTROL_DIR = Path("/tmp/sync_remote_runs_ssh")  # kept for cleanup on startup
SSH_CMD = (
    "ssh"
    " -o ControlMaster=no"            # do NOT share connections
    " -o ServerAliveInterval=15"
    " -o ServerAliveCountMax=3"
    " -o ConnectTimeout=15"
)

RSYNC_BASE_ARGS = [
    "rsync",
    "-ahz",              # archive, human-readable, compress (no -v; --info replaces)
    # Rsync 3.x progress reporting. progress2 = one global transfer
    # line (bytes, rate, ETA); name0 = current file name; stats2 =
    # post-transfer summary; flist2 = file-list enumeration during
    # handshake. Survives remote MOTD / module-load banner noise.
    "--info=progress2,name0,stats2,flist2",
    "--partial",
    "--no-motd",         # suppress rsync daemon MOTD banner
    "--timeout=120",     # generous IO timeout for flaky links
    "--partial-dir=.rsync-partial",  # keep partial files for resume
    "--exclude=__pycache__/",
    "--exclude=*.pyc",
    "-e", SSH_CMD,
]

MAX_RETRIES = 10         # retries per sync attempt
RETRY_BACKOFF_BASE = 5   # seconds, doubles each retry (5, 10, 20, ...)
RETRY_BACKOFF_MAX = 120  # cap backoff at 2 minutes


def ticket_is_valid() -> bool:
    """Check if we have a valid (non-expired) Kerberos ticket."""
    try:
        result = subprocess.run(
            [KLIST_BIN, "-s"],
            capture_output=True,
            timeout=5,
        )
        return result.returncode == 0
    except (FileNotFoundError, subprocess.TimeoutExpired):
        return False


def ensure_ticket() -> bool:
    """
    Ensure we have a valid Kerberos ticket. Only prompts for PIN
    if the current ticket is expired or missing.

    Returns True if ticket is valid after this call.
    """
    if ticket_is_valid():
        print("✓ Kerberos ticket is valid")
        return True

    print("⚠ Kerberos ticket expired or missing — running pkinit...")
    print(f"  {PKINIT_BIN} {PRINCIPAL}")
    print("  (You may need your smartcard + PIN)\n")

    result = subprocess.run([PKINIT_BIN, PRINCIPAL])
    if result.returncode != 0:
        print("✗ pkinit failed")
        return False

    if ticket_is_valid():
        print("✓ Kerberos ticket obtained")
        # Show ticket details for debugging
        subprocess.run([KLIST_BIN])
        return True

    print("✗ pkinit succeeded but ticket not valid — check klist")
    return False


def test_ssh_connection(remote_host: str | None = None) -> bool:
    """Quick SSH test to verify Kerberos auth actually works end-to-end."""
    if remote_host is None:
        remote_host = REMOTES[DEFAULT_REMOTE]["host"]
    print("  Testing SSH connection...", end=" ", flush=True)
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         remote_host,
         "echo ok"],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if result.returncode == 0 and "ok" in result.stdout:
        print("✓")
        return True
    print("✗")
    print(f"  SSH stderr: {result.stderr.strip()}")
    return False


def run_rsync(remote_path: str, local_path: Path, dry_run: bool = False,
              remote_host: str | None = None,
              best_only: bool = False) -> bool:
    """
    Run rsync with retry logic. Returns True on success.

    When ``best_only`` is True, only ``best.pt`` files are transferred
    (with the surrounding directory structure preserved).
    """
    if remote_host is None:
        remote_host = REMOTES[DEFAULT_REMOTE]["host"]
    local_path.mkdir(parents=True, exist_ok=True)

    args = list(RSYNC_BASE_ARGS)
    if best_only:
        # Descend into every directory, include best.pt, exclude
        # everything else. Order matters: rsync applies the first
        # matching rule per path.
        args += ["--include=*/", "--include=best.pt", "--exclude=*"]
    if dry_run:
        args.append("--dry-run")
    args += [f"{remote_host}:{remote_path}", str(local_path) + "/"]

    for attempt in range(1, MAX_RETRIES + 1):
        print(f"\n{'─' * 60}")
        print(f"rsync attempt {attempt}/{MAX_RETRIES}")
        print(f"  {remote_path} → {local_path}/")
        print(f"{'─' * 60}")

        result = subprocess.run(args)

        if result.returncode == 0:
            print(f"\n✓ rsync completed successfully")
            return True

        # rsync exit codes: 12=socket, 23=partial, 24=vanished, 30=timeout, 35=connection
        retriable = {10, 11, 12, 23, 24, 30, 35, 255}
        if result.returncode not in retriable:
            print(f"\n✗ rsync failed with non-retriable exit code {result.returncode}")
            return False

        # 255 = SSH auth failure — force re-auth even if klist -s says OK
        # (ticket can appear valid but be rejected by the remote host)
        if result.returncode == 255 or not ticket_is_valid():
            print("\n⚠ SSH auth failure or ticket issue — forcing re-authentication...")
            import os
            print(f"  KRB5CCNAME={os.environ.get('KRB5CCNAME', '(not set)')}")
            # Destroy the old (possibly stale) ticket first
            subprocess.run([KDESTROY_BIN], capture_output=True)
            if not ensure_ticket():
                print("✗ Cannot re-authenticate, aborting")
                return False
            # Verify SSH actually works with the new ticket
            if not test_ssh_connection(remote_host=remote_host):
                print("✗ SSH still failing after re-auth — credential cache issue?")
                print(f"  Try: export KRB5CCNAME=$(klist 2>&1 | grep 'cache:' | awk '{{print $NF}}')")
                return False

        backoff = min(RETRY_BACKOFF_BASE * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX)
        print(f"\n⚠ rsync exited with code {result.returncode}, retrying in {backoff}s...")
        time.sleep(backoff)

    print(f"\n✗ Failed after {MAX_RETRIES} attempts")
    return False


def sync_once(run_name: str | None = None, dry_run: bool = False,
              remote_name: str = DEFAULT_REMOTE,
              bootstrap: bool = False,
              best_only: bool = False) -> bool:
    """Run a single sync cycle. Returns True on success.

    When ``bootstrap`` is True, syncs the full ``bootstrap_runs/`` tree
    (or a specific bootstrap run id when ``run_name`` is given) instead
    of the regular ``runs/`` tree. This preserves the nested directory
    structure expected by ``compose_bootstrap_agent.py``.
    """
    # Wipe any stale multiplexing sockets left over from earlier
    # versions of this script — they can silently stall new rsyncs.
    if SSH_CONTROL_DIR.exists():
        import shutil as _shutil
        try:
            _shutil.rmtree(SSH_CONTROL_DIR)
        except OSError:
            pass

    if not ensure_ticket():
        return False

    remote_cfg = REMOTES[remote_name]
    remote_host = remote_cfg["host"]

    if bootstrap:
        remote_base = remote_cfg["bootstrap_runs"]
        local_base = LOCAL_BOOTSTRAP_RUNS
    else:
        remote_base = remote_cfg["runs"]
        local_base = LOCAL_RUNS

    if run_name:
        remote = f"{remote_base.rstrip('/')}/{run_name}/"
        local = local_base / run_name
    else:
        remote = remote_base
        local = local_base

    return run_rsync(remote, local, dry_run=dry_run, remote_host=remote_host,
                     best_only=best_only)


def main():
    parser = argparse.ArgumentParser(
        description="Sync training runs from HPC (with pkinit auth + retry)"
    )
    parser.add_argument(
        "--run",
        type=str,
        default=None,
        help="Sync a specific run directory (default: all runs)",
    )
    parser.add_argument(
        "--interval",
        type=int,
        default=None,
        help="Poll interval in seconds (default: sync once and exit)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be transferred without actually copying",
    )
    parser.add_argument(
        "--remote",
        type=str,
        default=DEFAULT_REMOTE,
        choices=list(REMOTES.keys()),
        help=f"Remote HPC to sync from (default: {DEFAULT_REMOTE})",
    )
    parser.add_argument(
        "--bootstrap",
        action="store_true",
        help="Sync the full bootstrap_runs/ tree instead of runs/. "
             "Preserves the nested layout expected by "
             "compose_bootstrap_agent.py. Combine with --run to sync "
             "a single bootstrap run id.",
    )
    parser.add_argument(
        "--best-only",
        action="store_true",
        help="Transfer only best.pt checkpoint files, preserving the "
             "enclosing directory structure. Useful for composition / "
             "evaluation when full training artifacts are not needed.",
    )
    args = parser.parse_args()

    remote_cfg = REMOTES[args.remote]
    remote_path = (remote_cfg["bootstrap_runs"] if args.bootstrap
                   else remote_cfg["runs"])
    print(f"Remote: {args.remote} ({remote_cfg['host']})")
    print(f"Path:   {remote_path}\n")

    if args.interval is not None:
        print(f"Polling every {args.interval}s (Ctrl-C to stop)\n")
        while True:
            try:
                sync_once(run_name=args.run, dry_run=args.dry_run,
                          remote_name=args.remote,
                          bootstrap=args.bootstrap,
                          best_only=args.best_only)
                print(f"\nSleeping {args.interval}s until next sync...")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                sys.exit(0)
    else:
        ok = sync_once(run_name=args.run, dry_run=args.dry_run,
                       remote_name=args.remote,
                       bootstrap=args.bootstrap,
                       best_only=args.best_only)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
