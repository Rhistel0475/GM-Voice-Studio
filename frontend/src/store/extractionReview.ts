import { create } from "zustand";
import type {
  ExtractionBatchResult,
  ExtractionConfidence,
  ExtractionEntity,
  ExtractionReviewItem,
  ExtractionReviewStatus,
} from "../types";
import { createId, nowIso } from "../lib/utils/ids";

export interface ExtractionReviewQueueState {
  items: ExtractionReviewItem[];
  /** When true, high-confidence items are auto-marked approved when enqueued. */
  autoApproveHighConfidence: boolean;
}

export interface ExtractionReviewQueueActions {
  /** Enqueue all entities from a batch into the review queue. */
  enqueueBatch: (batch: ExtractionBatchResult) => void;
  /** Update the review status of a specific item. */
  updateItemStatus: (id: string, status: ExtractionReviewStatus) => void;
  /** Apply edits to the underlying entity for a review item (UI-side only). */
  editItemEntity: (id: string, updater: (entity: ExtractionEntity) => ExtractionEntity) => void;
  /** Remove a single item from the queue. */
  removeItem: (id: string) => void;
  /** Clear all items from the queue. */
  clearQueue: () => void;
  /** Configure whether high-confidence items are auto-approved on enqueue. */
  setAutoApproveHighConfidence: (value: boolean) => void;
}

const initialState: ExtractionReviewQueueState = {
  items: [],
  autoApproveHighConfidence: true,
};

function deriveInitialStatus(
  confidence: ExtractionConfidence,
  existingStatus: ExtractionReviewStatus,
  autoApproveHigh: boolean
): { status: ExtractionReviewStatus; autoApproved: boolean } {
  // Respect existing explicit statuses first.
  if (existingStatus === "approved" || existingStatus === "rejected") {
    return { status: existingStatus, autoApproved: false };
  }

  if (confidence === "high") {
    if (autoApproveHigh) {
      return { status: "approved", autoApproved: true };
    }
    // Default to pending when not auto-approving.
    return { status: "pending", autoApproved: false };
  }

  // Medium/low confidence default to needs_review.
  if (confidence === "medium" || confidence === "low") {
    return { status: "needs_review", autoApproved: false };
  }

  return { status: existingStatus, autoApproved: false };
}

export const useExtractionReviewQueueStore = create<
  ExtractionReviewQueueState & ExtractionReviewQueueActions
>((set) => ({
  ...initialState,

  enqueueBatch(batch) {
    set((state) => {
      const now = nowIso();
      const nextItems: ExtractionReviewItem[] = batch.entities.map((entity) => {
        const { status, autoApproved } = deriveInitialStatus(
          entity.confidence,
          entity.reviewStatus,
          state.autoApproveHighConfidence
        );
        return {
          id: createId("review"),
          entity,
          confidence: entity.confidence,
          reviewStatus: status,
          source: entity.source,
          createdAt: batch.extractedAt || now,
          autoApproved,
        };
      });

      return {
        items: [...state.items, ...nextItems],
      };
    });
  },

  updateItemStatus(id, status) {
    set((state) => ({
      items: state.items.map((item) =>
        item.id === id ? { ...item, reviewStatus: status, autoApproved: false } : item
      ),
    }));
  },

  editItemEntity(id, updater) {
    set((state) => ({
      items: state.items.map((item) =>
        item.id === id ? { ...item, entity: updater(item.entity) } : item
      ),
    }));
  },

  removeItem(id) {
    set((state) => ({
      items: state.items.filter((item) => item.id !== id),
    }));
  },

  clearQueue() {
    set({ items: [] });
  },

  setAutoApproveHighConfidence(value) {
    set({ autoApproveHighConfidence: value });
  },
}));
