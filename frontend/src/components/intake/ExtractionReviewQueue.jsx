/**
 * ExtractionReviewQueue — review panel for entities extracted from a parsed document.
 *
 * Reads from useExtractionReviewQueueStore; on "Apply Approved" writes to the
 * campaign context store via applyApprovedEntities.
 */
import { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useExtractionReviewQueueStore } from "../../store/extractionReview";
import { useCampaignContextStore } from "../../store/campaignContext";
import { applyApprovedEntities } from "../../lib/applyApprovedEntities";

// ── Constants ─────────────────────────────────────────────────────────────────

const TYPE_LABELS = {
  npc: "NPC",
  location: "Location",
  scene_seed: "Scene",
  codex_entry: "Codex",
  encounter: "Encounter",
  item: "Item",
  faction: "Faction",
};

const TYPE_COLORS = {
  npc: "rq-badge--npc",
  location: "rq-badge--location",
  scene_seed: "rq-badge--scene",
  codex_entry: "rq-badge--codex",
  encounter: "rq-badge--encounter",
  item: "rq-badge--item",
  faction: "rq-badge--faction",
};

const CONFIDENCE_COLORS = {
  high: "rq-conf--high",
  medium: "rq-conf--medium",
  low: "rq-conf--low",
};

const STATUS_LABELS = {
  pending: "Pending",
  approved: "Approved",
  rejected: "Rejected",
  needs_review: "Needs Review",
};

const ALL_STATUSES = ["all", "pending", "needs_review", "approved", "rejected"];

// ── Helpers ───────────────────────────────────────────────────────────────────

function entityName(entity) {
  return entity.name || entity.title || "(unnamed)";
}

function entitySummary(entity) {
  return entity.summary || entity.description || "";
}

// ── Sub-components ────────────────────────────────────────────────────────────

function ReviewItem({ item, onApprove, onReject, onEdit }) {
  const [editing, setEditing] = useState(false);
  const [draftName, setDraftName] = useState(entityName(item.entity));
  const [draftSummary, setDraftSummary] = useState(entitySummary(item.entity));

  const handleSaveEdit = () => {
    onEdit(item.id, (entity) => {
      const updated = { ...entity };
      if ("name" in updated) updated.name = draftName;
      if ("title" in updated) updated.title = draftName;
      if ("summary" in updated) updated.summary = draftSummary;
      if ("description" in updated) updated.description = draftSummary;
      return updated;
    });
    setEditing(false);
  };

  const isApproved = item.reviewStatus === "approved";
  const isRejected = item.reviewStatus === "rejected";

  return (
    <div className={`rq-item ${isApproved ? "rq-item--approved" : ""} ${isRejected ? "rq-item--rejected" : ""}`}>
      <div className="rq-item-header">
        <span className={`rq-badge ${TYPE_COLORS[item.entity.type] || ""}`}>
          {TYPE_LABELS[item.entity.type] || item.entity.type}
        </span>
        <span className="rq-item-name">{entityName(item.entity)}</span>
        <span className={`rq-conf ${CONFIDENCE_COLORS[item.confidence] || ""}`}>
          {item.confidence}
        </span>
        <span className={`rq-status rq-status--${item.reviewStatus}`}>
          {STATUS_LABELS[item.reviewStatus]}
        </span>
        {item.autoApproved && (
          <span className="rq-auto-tag">auto</span>
        )}
      </div>

      {!editing && (
        <>
          {entitySummary(item.entity) && (
            <div className="rq-item-summary">{entitySummary(item.entity)}</div>
          )}
          {item.source?.excerpt && (
            <div className="rq-source">
              <span className="rq-source-label">Source:</span>
              {item.source.sectionHeading && (
                <span className="rq-source-section"> {item.source.sectionHeading} — </span>
              )}
              <span className="rq-source-excerpt">{item.source.excerpt}</span>
            </div>
          )}
        </>
      )}

      {editing && (
        <div className="rq-edit-form">
          <label className="rq-edit-label">Name / Title</label>
          <input
            className="rq-edit-input"
            value={draftName}
            onChange={(e) => setDraftName(e.target.value)}
          />
          <label className="rq-edit-label">Summary</label>
          <textarea
            className="rq-edit-textarea"
            value={draftSummary}
            onChange={(e) => setDraftSummary(e.target.value)}
            rows={3}
          />
          <div className="rq-edit-actions">
            <button type="button" className="rq-btn rq-btn--save" onClick={handleSaveEdit}>Save</button>
            <button type="button" className="rq-btn rq-btn--cancel" onClick={() => setEditing(false)}>Cancel</button>
          </div>
        </div>
      )}

      <div className="rq-item-actions">
        {!isApproved && (
          <button
            type="button"
            className="rq-btn rq-btn--approve"
            onClick={() => onApprove(item.id)}
          >
            Approve
          </button>
        )}
        {!isRejected && (
          <button
            type="button"
            className="rq-btn rq-btn--reject"
            onClick={() => onReject(item.id)}
          >
            Reject
          </button>
        )}
        {isApproved && (
          <button
            type="button"
            className="rq-btn rq-btn--pending"
            onClick={() => onApprove(item.id, "pending")}
          >
            Undo
          </button>
        )}
        {isRejected && (
          <button
            type="button"
            className="rq-btn rq-btn--pending"
            onClick={() => onReject(item.id, "pending")}
          >
            Undo
          </button>
        )}
        {!editing && (
          <button
            type="button"
            className="rq-btn rq-btn--edit"
            onClick={() => setEditing(true)}
          >
            Edit
          </button>
        )}
      </div>
    </div>
  );
}

// ── Main component ────────────────────────────────────────────────────────────

/**
 * @param {{ documentName?: string, onNavigate?: (view: string) => void }} props
 */
export default function ExtractionReviewQueue({ documentName, onNavigate }) {
  const navigate = useNavigate();
  const {
    items,
    autoApproveHighConfidence,
    updateItemStatus,
    editItemEntity,
    clearQueue,
    setAutoApproveHighConfidence,
  } = useExtractionReviewQueueStore();

  const storeState = useCampaignContextStore();

  const [filterStatus, setFilterStatus] = useState("all");
  const [applyResult, setApplyResult] = useState(null);
  const [applyError, setApplyError] = useState("");

  const filtered = filterStatus === "all"
    ? items
    : items.filter((i) => i.reviewStatus === filterStatus);

  const approvedCount = items.filter((i) => i.reviewStatus === "approved").length;
  const rejectedCount = items.filter((i) => i.reviewStatus === "rejected").length;
  const pendingCount = items.filter(
    (i) => i.reviewStatus === "pending" || i.reviewStatus === "needs_review"
  ).length;

  const handleApprove = (id, forcedStatus) => {
    updateItemStatus(id, forcedStatus ?? "approved");
  };

  const handleReject = (id, forcedStatus) => {
    updateItemStatus(id, forcedStatus ?? "rejected");
  };

  const handleApproveAll = () => {
    filtered.forEach((item) => {
      if (item.reviewStatus !== "approved") updateItemStatus(item.id, "approved");
    });
  };

  const handleRejectAll = () => {
    filtered.forEach((item) => {
      if (item.reviewStatus !== "rejected") updateItemStatus(item.id, "rejected");
    });
  };

  const handleApplyApproved = () => {
    setApplyError("");
    setApplyResult(null);
    try {
      const result = applyApprovedEntities(
        useCampaignContextStore.getState().npcs.length >= 0
          ? useExtractionReviewQueueStore.getState().items
          : items,
        {
          upsertNpc: storeState.upsertNpc,
          upsertScene: storeState.upsertScene,
          upsertCodexEntry: storeState.upsertCodexEntry,
          upsertCampaign: storeState.upsertCampaign,
          setActiveCampaign: storeState.setActiveCampaign,
          activeCampaignId: storeState.activeCampaignId,
          campaigns: storeState.campaigns,
        },
        documentName || "Imported Campaign"
      );
      setApplyResult(result);
    } catch (err) {
      setApplyError(err?.message || "Failed to apply entities.");
    }
  };

  if (items.length === 0) {
    return (
      <div className="rq-empty">
        <p>No entities in the review queue.</p>
        <p className="rq-empty-hint">Parse a document above, then come back here to review extracted entities before adding them to your campaign.</p>
      </div>
    );
  }

  return (
    <div className="rq-shell">
      {/* Header bar */}
      <div className="rq-header">
        <div className="rq-stats">
          <span className="rq-stat rq-stat--total">{items.length} entities</span>
          <span className="rq-stat rq-stat--approved">{approvedCount} approved</span>
          <span className="rq-stat rq-stat--pending">{pendingCount} pending</span>
          <span className="rq-stat rq-stat--rejected">{rejectedCount} rejected</span>
        </div>
        <label className="rq-auto-label">
          <input
            type="checkbox"
            checked={autoApproveHighConfidence}
            onChange={(e) => setAutoApproveHighConfidence(e.target.checked)}
          />
          {" "}Auto-approve high confidence
        </label>
      </div>

      {/* Filter strip */}
      <div className="rq-filter-strip">
        {ALL_STATUSES.map((s) => (
          <button
            key={s}
            type="button"
            className={`rq-filter-btn ${filterStatus === s ? "rq-filter-btn--active" : ""}`}
            onClick={() => setFilterStatus(s)}
          >
            {s === "all" ? `All (${items.length})` : STATUS_LABELS[s]}
          </button>
        ))}
      </div>

      {/* Batch actions */}
      <div className="rq-batch-actions">
        <button type="button" className="rq-btn rq-btn--approve" onClick={handleApproveAll}>
          Approve All Shown
        </button>
        <button type="button" className="rq-btn rq-btn--reject" onClick={handleRejectAll}>
          Reject All Shown
        </button>
        <button type="button" className="rq-btn rq-btn--clear" onClick={clearQueue}>
          Clear Queue
        </button>
        <button
          type="button"
          className={`rq-btn rq-btn--apply ${approvedCount === 0 ? "rq-btn--disabled" : ""}`}
          onClick={handleApplyApproved}
          disabled={approvedCount === 0}
        >
          Apply {approvedCount > 0 ? `${approvedCount} Approved` : "Approved"} to Campaign
        </button>
      </div>

      {/* Apply result / error feedback */}
      {applyResult && (
        <div className="rq-apply-result">
          <div>
            Applied to campaign: {applyResult.npcCount} NPC{applyResult.npcCount !== 1 ? "s" : ""},
            {" "}{applyResult.sceneCount} scene{applyResult.sceneCount !== 1 ? "s" : ""},
            {" "}{applyResult.codexCount} codex entr{applyResult.codexCount !== 1 ? "ies" : "y"}.
          </div>
          <div className="rq-apply-nav">
            <button
              type="button"
              className="rq-btn rq-btn--nav"
              onClick={() => onNavigate ? onNavigate("codex") : navigate("/codex")}
            >
              View in Campaign Codex →
            </button>
            <button
              type="button"
              className="rq-btn rq-btn--nav"
              onClick={() => onNavigate ? onNavigate("live") : navigate("/")}
            >
              Go to Live Board →
            </button>
          </div>
        </div>
      )}
      {applyError && (
        <div className="rq-apply-error">{applyError}</div>
      )}

      {/* Item list */}
      <div className="rq-list">
        {filtered.length === 0 && (
          <div className="rq-list-empty">No items match the current filter.</div>
        )}
        {filtered.map((item) => (
          <ReviewItem
            key={item.id}
            item={item}
            onApprove={handleApprove}
            onReject={handleReject}
            onEdit={editItemEntity}
          />
        ))}
      </div>
    </div>
  );
}
