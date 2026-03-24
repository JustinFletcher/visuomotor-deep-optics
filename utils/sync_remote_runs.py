#!/usr/bin/env python3
"""
Sync training runs from HPC to local machine.

Handles Kerberos/pkinit authentication (smartcard + PIN) and retries
dropped rsync connections automatically. Only re-authenticates when
the Kerberos ticket expires, so you enter your PIN once and it stays
valid for the ticket lifetime (typically 10-24 hours).

Usage:
    python utils/sync_remote_runs.py                    # sync all runs
    python utils/sync_remote_runs.py --run ppo_optomech_1_1774138836  # sync one run
    python utils/sync_remote_runs.py --interval 60      # poll every 60s
    python utils/sync_remote_runs.py --dry-run           # show what would transfer
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

REMOTE_HOST = "fletch@makau.mhpcc.hpc.mil"
REMOTE_RUNS = "/p/home/fletch/visuomotor-deep-optics/runs/"
LOCAL_RUNS = Path(__file__).resolve().parent.parent / "runs"

# SSH ControlMaster: keep a persistent connection open so we only
# authenticate once. All subsequent rsync calls multiplex over it.
SSH_CONTROL_DIR = Path("/tmp/sync_remote_runs_ssh")
SSH_CONTROL_PATH = SSH_CONTROL_DIR / "ctrl-%r@%h:%p"
SSH_CMD = (
    "ssh"
    " -o ControlMaster=auto"
    f" -o ControlPath={SSH_CONTROL_PATH}"
    " -o ControlPersist=600"          # keep socket alive 10 min after last use
    " -o ServerAliveInterval=15"
    " -o ServerAliveCountMax=3"
)

RSYNC_BASE_ARGS = [
    "rsync",
    "-havzP",            # human-readable, archive, verbose, compress, partial+progress
    "--stats",
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


def test_ssh_connection() -> bool:
    """Quick SSH test to verify Kerberos auth actually works end-to-end."""
    print("  Testing SSH connection...", end=" ", flush=True)
    result = subprocess.run(
        ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10",
         REMOTE_HOST.split("@")[1] if "@" not in REMOTE_HOST else REMOTE_HOST,
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


def run_rsync(remote_path: str, local_path: Path, dry_run: bool = False) -> bool:
    """
    Run rsync with retry logic. Returns True on success.
    """
    local_path.mkdir(parents=True, exist_ok=True)

    args = list(RSYNC_BASE_ARGS)
    if dry_run:
        args.append("--dry-run")
    args += [f"{REMOTE_HOST}:{remote_path}", str(local_path) + "/"]

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
            if not test_ssh_connection():
                print("✗ SSH still failing after re-auth — credential cache issue?")
                print(f"  Try: export KRB5CCNAME=$(klist 2>&1 | grep 'cache:' | awk '{{print $NF}}')")
                return False

        backoff = min(RETRY_BACKOFF_BASE * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX)
        print(f"\n⚠ rsync exited with code {result.returncode}, retrying in {backoff}s...")
        time.sleep(backoff)

    print(f"\n✗ Failed after {MAX_RETRIES} attempts")
    return False


def sync_once(run_name: str | None = None, dry_run: bool = False) -> bool:
    """Run a single sync cycle. Returns True on success."""
    SSH_CONTROL_DIR.mkdir(parents=True, exist_ok=True)
    if not ensure_ticket():
        return False

    if run_name:
        remote = f"{REMOTE_RUNS.rstrip('/')}/{run_name}/"
        local = LOCAL_RUNS / run_name
    else:
        remote = REMOTE_RUNS
        local = LOCAL_RUNS

    return run_rsync(remote, local, dry_run=dry_run)


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
    args = parser.parse_args()

    if args.interval is not None:
        print(f"Polling every {args.interval}s (Ctrl-C to stop)\n")
        while True:
            try:
                sync_once(run_name=args.run, dry_run=args.dry_run)
                print(f"\nSleeping {args.interval}s until next sync...")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nStopped.")
                sys.exit(0)
    else:
        ok = sync_once(run_name=args.run, dry_run=args.dry_run)
        sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
