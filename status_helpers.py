from dao_selector import DAOS

def set_enrichment_status(user_id, dataset_label, *,
                          status="running",
                          phase="init",
                          detail=None,
                          batches_done=0,
                          total_batches=None,
                          percent=None):
    """Upsert enrichment status into Cloudflare D1."""
    d1 = DAOS.get("main")
    if d1 is None:
        print("[warn] D1 DAO not configured; skipping enrichment status write.")
        return
    try:
        d1.upsert_enrichment_status(
            user_id=user_id,
            dataset_label=dataset_label,
            status=status,
            phase=phase,
            detail=detail,
            batches_done=batches_done,
            total_batches=total_batches,
            percent=percent,
        )
        print(f"[status] {user_id}/{dataset_label}: {phase} → {status}")
    except Exception as e:
        print(f"[status] ⚠️ Failed to update enrichment_status: {e}")

def finish_enrichment_status(user_id, dataset_label, ok, detail=None):
    """Mark enrichment complete/failed."""
    set_enrichment_status(
        user_id=user_id,
        dataset_label=dataset_label,
        status="completed" if ok else "failed",
        phase="done",
        detail=detail,
        percent=100 if ok else None,
    )
