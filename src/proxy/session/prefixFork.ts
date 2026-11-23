/**
 * Prefix-based fork source discovery and checkpoint rewind helpers.
 *
 * When an intermediary proxy (such as new-api or CLIProxyAPI) strips the
 * `x-meridian-fork-from` header, a newly keyed conversation or fork-summary
 * request arrives with no durable mapping of its own even though its parent
 * session is already published on the same profile. Matching against published
 * sessions on that profile by cryptographic message-hash prefix (and, when
 * present, exact tool-call ID settlement) lets the turn fork the parent's SDK
 * transcript and keep the prompt cache warm.
 */

import { readSessionStoreSnapshot, type StoredSession } from "../sessionStore"
import {
  coalesceCompleteToolResultContinuation,
  settlesCheckpointThenContinues,
} from "../passthroughEarlyStop"
import type { ToolCallInfo } from "../messages"
import { flattenAssistantContent, normalizeStructuredUserContent } from "../replay"
import {
  computeMessageHashes,
  verifyLineage,
  type SessionState,
} from "./lineage"
import { messagesHeld } from "./cache"

export interface PrefixForkCandidateOptions {
  readonly targetKey: string
  readonly profileId: string
  readonly messages: Array<{ role: string; content: any }>
  readonly expectedTranscript?: { readonly configDir: string; readonly projectDir?: string }
  readonly resumeContextHash?: string
  readonly allowTrailingSystemReminder?: boolean
  /** Minimum stored prefix length when no tool-call ID settlement proves the link. */
  readonly minPrefixMessages?: number
  /** Require strictly at least this many matched leading messages (used when rebinding a stale mapping). */
  readonly minMatchedMessages?: number
  /** Optional snapshot override for deterministic unit tests. */
  readonly snapshot?: Record<string, StoredSession>
}

export interface PrefixForkMatch {
  readonly key: string
  readonly session: StoredSession
  readonly matchedMessages: number
  readonly lineageType: "continuation" | "undo"
}

export type CheckpointRewindPlan =
  | {
    readonly kind: "tool_checkpoint"
    readonly assistantUuid: string
    readonly toolIds: string[]
    readonly messagesToConvert: Array<{ role: "user"; content: unknown[] }>
    readonly prefixLen: number
  }
  | {
    readonly kind: "undo_checkpoint"
    readonly assistantUuid: string
    readonly messagesToConvert: Array<{ role: string; content: any }>
    readonly prefixLen: number
  }

/** Extract all Anthropic `tool_use` IDs from an assistant message. */
export function extractAssistantToolUseIds(
  message: { role?: unknown; content?: unknown } | undefined,
): string[] {
  if (!message || message.role !== "assistant" || !Array.isArray(message.content)) return []
  const ids: string[] = []
  for (const raw of message.content) {
    const block = raw as { type?: unknown; id?: unknown } | null | undefined
    if (block?.type === "tool_use" && typeof block.id === "string" && block.id.length > 0) {
      ids.push(block.id)
    }
  }
  return ids
}

function sessionStateFromStored(shared: StoredSession): SessionState {
  return {
    claudeSessionId: shared.claudeSessionId,
    lastAccess: shared.lastUsedAt,
    messageCount: shared.messageCount || 0,
    lineageHash: shared.lineageHash || "",
    clientLineageHash: shared.clientLineageHash,
    resumeContextHash: shared.resumeContextHash,
    messageHashes: shared.messageHashes,
    messageBlockHashes: shared.messageBlockHashes,
    sdkMessageUuids: shared.sdkMessageUuids,
    passthroughToolCallAssistantUuid: shared.passthroughToolCallAssistantUuid,
    passthroughToolCallIds: shared.passthroughToolCallIds,
    contextUsage: shared.contextUsage,
    currentTranscript: shared.currentTranscript,
    previousTranscript: shared.previousTranscript,
  }
}

function hasLegacyUserDenialBoundary(session: StoredSession): boolean {
  const legacy = (session as StoredSession & { passthroughResumeUuid?: unknown }).passthroughResumeUuid
  return typeof legacy === "string" && legacy.length > 0 && !session.passthroughToolCallAssistantUuid
}

function matchesExpectedTranscript(
  session: StoredSession,
  expected: { readonly configDir: string; readonly projectDir?: string } | undefined,
): boolean {
  if (!session.currentTranscript) return true
  if (session.currentTranscript.sessionId !== session.claudeSessionId) return false
  if (!expected) return true
  if (session.currentTranscript.configDir !== expected.configDir) return false
  if (expected.projectDir !== undefined && session.currentTranscript.projectDir !== expected.projectDir) return false
  return true
}

/**
 * Find the best published session on the same profile whose history is a valid
 * prefix of `messages`, so a newly keyed fork (or a stale mapping whose
 * fallback key moved ahead) can resume via managed fork without replaying cold.
 */
export function findForkSourceByPrefix(options: PrefixForkCandidateOptions): PrefixForkMatch | undefined {
  const { targetKey, profileId, messages, expectedTranscript, resumeContextHash } = options
  if (messages.length < 2) return undefined

  const minPrefix = options.minPrefixMessages ?? 3
  const minMatched = options.minMatchedMessages ?? 1
  const reminderOpts = { allowTrailingSystemReminder: options.allowTrailingSystemReminder === true }
  const incomingHashes = computeMessageHashes(messages)
  const firstHash = incomingHashes[0]
  if (!firstHash) return undefined

  const snapshot = options.snapshot ?? readSessionStoreSnapshot()
  const profilePrefix = profileId !== "default" ? `${profileId}:` : undefined

  let best: PrefixForkMatch | undefined
  const isBetter = (candidate: PrefixForkMatch): boolean => {
    if (!best) return true
    if (candidate.matchedMessages !== best.matchedMessages) {
      return candidate.matchedMessages > best.matchedMessages
    }
    if (candidate.lineageType !== best.lineageType) {
      return candidate.lineageType === "continuation"
    }
    return candidate.session.lastUsedAt > best.session.lastUsedAt
  }

  for (const [candidateKey, session] of Object.entries(snapshot)) {
    if (candidateKey === targetKey) continue
    if (profilePrefix && !candidateKey.startsWith(profilePrefix)) continue
    if (!session || !session.claudeSessionId || !session.lineageHash) continue
    if (hasLegacyUserDenialBoundary(session)) continue
    if (!matchesExpectedTranscript(session, expectedTranscript)) continue
    if (resumeContextHash && session.resumeContextHash && session.resumeContextHash !== resumeContextHash) continue
    if (!Array.isArray(session.messageHashes) || session.messageHashes.length === 0) continue
    if (session.messageHashes[0] !== firstHash) continue

    const state = sessionStateFromStored(session)
    const lineage = verifyLineage(state, messages, { compactionSurvival: false })

    if (lineage.type === "continuation" && lineage.resumeContentFrom === undefined) {
      const delta = messages.slice(lineage.resumeFrom)
      const pendingIds = session.passthroughToolCallIds
      if (pendingIds && pendingIds.length > 0) {
        const settled = Boolean(
          coalesceCompleteToolResultContinuation(delta, pendingIds, reminderOpts)
          || settlesCheckpointThenContinues(delta, pendingIds, reminderOpts),
        )
        if (!settled) continue
      } else if (session.messageCount < minPrefix) {
        continue
      }
      const matchedMessages = messagesHeld(lineage, messages)
      if (matchedMessages < minMatched) continue
      const candidate: PrefixForkMatch = { key: candidateKey, session, matchedMessages, lineageType: "continuation" }
      if (isBetter(candidate)) best = candidate
      continue
    }

    if (lineage.type === "undo" && lineage.rollbackUuid) {
      const rewindIdx = lineage.prefixOverlap - 1
      if (rewindIdx < 0) continue
      const rewindDelta = messages.slice(lineage.prefixOverlap)
      if (rewindDelta.length === 0) continue
      const rewindToolIds = extractAssistantToolUseIds(messages[rewindIdx])
      if (rewindToolIds.length > 0) {
        if (!coalesceCompleteToolResultContinuation(rewindDelta, rewindToolIds, reminderOpts)) continue
      } else {
        if (lineage.prefixOverlap < minPrefix) continue
        if (!rewindDelta.every((m) => m.role === "user" || m.role === "system")) continue
      }
      const matchedMessages = lineage.prefixOverlap
      if (matchedMessages < minMatched) continue
      const candidate: PrefixForkMatch = { key: candidateKey, session, matchedMessages, lineageType: "undo" }
      if (isBetter(candidate)) best = candidate
    }
  }

  return best
}

/**
 * When a passthrough tool checkpoint is unsettled (e.g. an interrupted run
 * discarded the last assistant turn and appended user text to the preceding
 * user message), check whether we can rewind to the immediately preceding
 * assistant UUID (`messageCount - 2`) and re-send only the updated trailing
 * user turn instead of discarding the entire cached conversation.
 */
export function resolveCheckpointRewind(input: {
  readonly cachedSession: SessionState | undefined
  readonly resumeFrom: number | undefined
  readonly allMessages: Array<{ role: string; content: any }>
  readonly allowTrailingSystemReminder: boolean
}): CheckpointRewindPlan | undefined {
  const { cachedSession, resumeFrom, allMessages, allowTrailingSystemReminder } = input
  if (!cachedSession || cachedSession.messageCount < 2 || resumeFrom === undefined) return undefined
  const prevAssistantIdx = cachedSession.messageCount - 2
  if (resumeFrom < prevAssistantIdx + 1) return undefined

  const prevAssistantUuid = cachedSession.sdkMessageUuids?.[prevAssistantIdx]
  if (typeof prevAssistantUuid !== "string" || prevAssistantUuid.length === 0) return undefined
  if (allMessages[prevAssistantIdx]?.role !== "assistant") return undefined
  if (allMessages[prevAssistantIdx + 1]?.role !== "user") return undefined

  const rewindDelta = allMessages.slice(prevAssistantIdx + 1)
  if (rewindDelta.length === 0) return undefined

  const prevToolIds = extractAssistantToolUseIds(allMessages[prevAssistantIdx])
  if (prevToolIds.length > 0) {
    const coalesced = coalesceCompleteToolResultContinuation(
      rewindDelta,
      prevToolIds,
      { allowTrailingSystemReminder },
    )
    if (!coalesced) return undefined
    return {
      kind: "tool_checkpoint",
      assistantUuid: prevAssistantUuid,
      toolIds: prevToolIds,
      messagesToConvert: coalesced,
      prefixLen: prevAssistantIdx + 1,
    }
  }

  if (!rewindDelta.every((m) => m.role === "user" || m.role === "system")) return undefined
  return {
    kind: "undo_checkpoint",
    assistantUuid: prevAssistantUuid,
    messagesToConvert: rewindDelta,
    prefixLen: prevAssistantIdx + 1,
  }
}

/**
 * Build a single structured user turn when rolling back to an assistant `tool_use`
 * boundary (`prefixOverlap - 1`). Preserves the structured `tool_result` blocks in
 * `tail[0]` that settle `expectedToolIds`, and flattens any preserved trailing turns
 * (`tail.slice(1)`, such as surviving turns after a middle `context_replace`) into
 * replay text/media blocks so no orphaned `tool_result` blocks reach the API.
 */
export function buildToolUndoRewindMessages(input: {
  readonly tail: Array<{ role: string; content: any }>
  readonly expectedToolIds: readonly string[]
  readonly allowTrailingSystemReminder: boolean
  readonly toolIndex?: Map<string, ToolCallInfo>
}): Array<{ role: "user"; content: unknown[] }> | undefined {
  const { tail, expectedToolIds, allowTrailingSystemReminder, toolIndex } = input
  if (tail.length === 0 || expectedToolIds.length === 0) return undefined
  const reminderOpts = { allowTrailingSystemReminder }
  const direct = coalesceCompleteToolResultContinuation(tail, expectedToolIds, reminderOpts)
  if (direct) return direct

  if (tail.length === 1) return undefined
  const first = coalesceCompleteToolResultContinuation([tail[0]!], expectedToolIds, reminderOpts)
  if (!first) return undefined

  const content: unknown[] = [...first[0]!.content]
  for (const message of tail.slice(1)) {
    if (message.role === "assistant") {
      const assistantText = flattenAssistantContent(message.content)
      if (assistantText) {
        content.push({ type: "text", text: `[Assistant: ${assistantText}]` })
      }
      continue
    }
    const normalized = normalizeStructuredUserContent(message.content, false, toolIndex)
    if (typeof normalized === "string") {
      if (normalized.length > 0) content.push({ type: "text", text: normalized })
    } else if (Array.isArray(normalized)) {
      content.push(...normalized)
    }
  }
  return [{ role: "user", content }]
}
