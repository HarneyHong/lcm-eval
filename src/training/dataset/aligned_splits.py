"""Shared manifest-based splits for aligned workload-driven experiments."""

import json

from training.dataset.plan_dataset import PlanDataset


PROTOCOLS = {"legacy", "baseline_native", "qpp_native", "matched"}


def plan_query_id(plan):
    query_id = getattr(plan, "query_id", None)
    if query_id is None:
        raise ValueError("Aligned workload plan is missing query_id")
    return str(query_id)


def load_aligned_ids(split_manifest_path, alignment_manifest_path, apply_qppnet_support):
    with open(split_manifest_path) as split_file:
        split = json.load(split_file)
    partitions = {name: [str(query_id) for query_id in split[name]]
                  for name in ("train", "validation", "test")}
    all_ids = partitions["train"] + partitions["validation"] + partitions["test"]
    if len(all_ids) != len(set(all_ids)):
        raise ValueError("Split manifest contains duplicate query IDs")

    if alignment_manifest_path is not None:
        with open(alignment_manifest_path) as alignment_file:
            alignment = json.load(alignment_file)
        master_ids = [str(query_id) for query_id in alignment["master_query_ids"]]
        if set(all_ids) != set(master_ids):
            raise ValueError("Split manifest does not partition the alignment master")
        if apply_qppnet_support:
            supported = {
                query_id for query_id in master_ids
                if alignment["queries"][query_id].get("qppnet_supported") is True
            }
            partitions = {
                name: [query_id for query_id in query_ids if query_id in supported]
                for name, query_ids in partitions.items()
            }
    elif apply_qppnet_support:
        raise ValueError("QPP-Net support filtering requires --alignment_manifest")
    return partitions


def create_aligned_datasets(plans, split_manifest_path, alignment_manifest_path=None,
                            apply_qppnet_support=False):
    plans_by_id = {}
    for plan in plans:
        query_id = plan_query_id(plan)
        if query_id in plans_by_id:
            raise ValueError(f"Duplicate query_id in workload: {query_id}")
        plans_by_id[query_id] = plan

    partitions = load_aligned_ids(
        split_manifest_path,
        alignment_manifest_path,
        apply_qppnet_support=apply_qppnet_support,
    )
    datasets = []
    for name in ("train", "validation", "test"):
        ids = partitions[name]
        missing = [query_id for query_id in ids if query_id not in plans_by_id]
        if missing:
            raise ValueError(f"Workload is missing {len(missing)} {name} query IDs; first={missing[0]}")
        datasets.append(PlanDataset([plans_by_id[query_id] for query_id in ids], ids))
    return (*datasets, partitions)
