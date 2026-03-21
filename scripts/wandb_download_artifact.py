"""
Download a Weights & Biases artifact using the public API (no wandb.init run).

Example:
  python -m scripts.wandb_download_artifact \\
    --artifact kwizzlesurp10-sevtech/nanochat/run-k9head2f-events:v0 \\
    --type wandb-events

Or split:
  python -m scripts.wandb_download_artifact \\
    --entity kwizzlesurp10-sevtech --project nanochat \\
    --name run-k9head2f-events:v0 --type wandb-events
"""

from __future__ import annotations

import argparse

import wandb

from nanochat.common import print0, resolve_wandb_artifact_full_name


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--artifact",
        default=None,
        help="Full path: entity/project/name:alias",
    )
    parser.add_argument("--entity", default=None, help="With --project and --name if --artifact omitted")
    parser.add_argument("--project", default=None)
    parser.add_argument(
        "--name",
        default=None,
        dest="name_with_alias",
        metavar="NAME",
        help="Artifact name and optional alias, e.g. run-abc-events:v0",
    )
    parser.add_argument(
        "--type",
        default=None,
        help="W&B artifact type if required (e.g. wandb-events)",
    )
    parser.add_argument(
        "--root",
        default=None,
        help="Directory root for download (default: W&B ./artifacts/...)",
    )
    args = parser.parse_args()

    full = resolve_wandb_artifact_full_name(
        args.artifact,
        entity=args.entity,
        project=args.project,
        name_with_alias=args.name_with_alias,
    )

    api = wandb.Api()
    artifact = api.artifact(full, type=args.type)
    artifact_dir = artifact.download(root=args.root)
    print0(artifact_dir)


if __name__ == "__main__":
    main()
