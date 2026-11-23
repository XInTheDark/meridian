import { beforeEach, describe, expect, it } from "bun:test"
import type { Options } from "@anthropic-ai/claude-agent-sdk"
import { installSdkMock } from "./sdkMock"
import { installLoggerMock } from "./loggerMock"
import { installMcpToolsMock } from "./mcpToolsMock"
import { assistantMessage, withMockSdkSessionId } from "./helpers"
import { buildToolUndoRewindMessages, findForkSourceByPrefix, resolveCheckpointRewind } from "../proxy/session/prefixFork"
import { computeMessageHashes, computeLineageHash, verifyLineage, type SessionState } from "../proxy/session/lineage"
import type { StoredSession } from "../proxy/sessionStore"

interface CapturedQuery {
  prompt: string | AsyncIterable<unknown>
  options: Options
}
const captured: Array<{ options: Options; prompt: string; rawMessages: unknown[] }> = []

installSdkMock(() => ({
  query: (params: CapturedQuery) => (async function* () {
    let prompt = ""
    const rawMessages: unknown[] = []
    if (typeof params.prompt === "string") {
      prompt = params.prompt
    } else {
      for await (const message of params.prompt) {
        rawMessages.push(message)
        prompt += JSON.stringify(message)
      }
    }
    captured.push({ options: params.options, prompt, rawMessages })
    yield withMockSdkSessionId(assistantMessage([{ type: "text", text: "acknowledged" }]), params.options)
  })(),
  createSdkMcpServer: () => ({ type: "sdk", name: "test", instance: { tool: () => {}, registerTool: () => ({}) } }),
  tool: () => ({}),
}), "session-prefix-fork-and-rewind.test.ts")
installLoggerMock(() => ({ claudeLog: () => {}, withClaudeLogContext: (_ctx: unknown, fn: () => unknown) => fn() }))
installMcpToolsMock(() => ({ createOpencodeMcpServer: () => ({ type: "sdk", name: "opencode", instance: {} }) }))

const { createProxyServer, clearSessionCache } = await import("../proxy/server")
const { lookupSharedSession, storeSharedSession } = await import("../proxy/sessionStore")

beforeEach(async () => {
  await clearSessionCache()
  captured.length = 0
})

function makeStoredSession(
  claudeSessionId: string,
  messages: Array<{ role: string; content: any }>,
  overrides: Partial<StoredSession> = {},
): StoredSession {
  return {
    claudeSessionId,
    createdAt: 1000,
    lastUsedAt: 2000,
    messageCount: messages.length,
    lineageHash: computeLineageHash(messages),
    messageHashes: computeMessageHashes(messages),
    sdkMessageUuids: messages.map((m, i) => (m.role === "assistant" ? `asst-uuid-${i}` : null)),
    ...overrides,
  }
}

describe("findForkSourceByPrefix (unit)", () => {
  it("matches a parent with pending passthroughToolCallIds even on turn 1 when the fork settles those exact IDs", () => {
    const parentMsgs = [{ role: "user", content: "Start task" }]
    const incoming = [
      { role: "user", content: "Start task" },
      { role: "assistant", content: [{ type: "tool_use", id: "toolu_split_1", name: "split", input: {} }] },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "toolu_split_1", content: "Preparing handoff" },
          { type: "text", text: "Write the handoff summary." },
        ],
      },
    ]
    const snapshot: Record<string, StoredSession> = {
      "claude1:parent-key": makeStoredSession("sdk-parent", parentMsgs, {
        passthroughToolCallAssistantUuid: "asst-split-uuid",
        passthroughToolCallIds: ["toolu_split_1"],
      }),
    }

    const match = findForkSourceByPrefix({
      targetKey: "claude1:fork-summary-key",
      profileId: "claude1",
      messages: incoming,
      allowTrailingSystemReminder: true,
      snapshot,
    })

    expect(match?.key).toBe("claude1:parent-key")
    expect(match?.lineageType).toBe("continuation")
  })

  it("refuses to cross-fork unrelated 1-message conversations without tool-call ID settlement", () => {
    const parentMsgs = [{ role: "user", content: "Hello" }]
    const incoming = [
      { role: "user", content: "Hello" },
      { role: "assistant", content: "Hi there!" },
      { role: "user", content: "How are you?" },
    ]
    const snapshot: Record<string, StoredSession> = {
      "claude1:other-chat": makeStoredSession("sdk-other", parentMsgs),
    }

    const match = findForkSourceByPrefix({
      targetKey: "claude1:new-chat",
      profileId: "claude1",
      messages: incoming,
      snapshot,
    })

    expect(match).toBeUndefined()
  })

  it("enforces profile isolation and never matches sessions from another profile", () => {
    const shared = [
      { role: "user", content: "Turn 1" },
      { role: "assistant", content: "Reply 1" },
      { role: "user", content: "Turn 2" },
    ]
    const incoming = [
      ...shared,
      { role: "assistant", content: "Reply 2" },
      { role: "user", content: "Lane brief" },
    ]
    const snapshot: Record<string, StoredSession> = {
      "claude2:parent-key": makeStoredSession("sdk-other-profile", shared),
    }

    const match = findForkSourceByPrefix({
      targetKey: "claude1:lane-key",
      profileId: "claude1",
      messages: incoming,
      snapshot,
    })

    expect(match).toBeUndefined()
  })

  it("matches an undo fork when the parent moved on past the split tool call", () => {
    const parentMsgs = [
      { role: "user", content: "Initial goal" },
      { role: "assistant", content: [{ type: "tool_use", id: "toolu_split_2", name: "split", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_split_2", content: "Started lanes A, B" }] },
      { role: "assistant", content: "Waiting for lanes." },
      { role: "user", content: "Parent follow-up." },
    ]
    const laneMsgs = [
      { role: "user", content: "Initial goal" },
      { role: "assistant", content: [{ type: "tool_use", id: "toolu_split_2", name: "split", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_split_2", content: "Lane B assignment" }] },
    ]
    const snapshot: Record<string, StoredSession> = {
      "claude1:parent-key": makeStoredSession("sdk-parent-advanced", parentMsgs),
    }

    const match = findForkSourceByPrefix({
      targetKey: "claude1:lane-b",
      profileId: "claude1",
      messages: laneMsgs,
      snapshot,
    })

    expect(match?.key).toBe("claude1:parent-key")
    expect(match?.lineageType).toBe("undo")
    expect(match?.matchedMessages).toBe(2)
  })
})

describe("resolveCheckpointRewind (unit)", () => {
  it("rewinds to the previous assistant tool_use UUID when a parallel tool-result user turn grows without echoing the dropped next turn", () => {
    const messages = [
      { role: "user", content: "Start" },
      { role: "assistant", content: [{ type: "tool_use", id: "toolu_a", name: "steer", input: {} }] },
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "toolu_a", content: "steered" },
          { type: "text", text: "Swarm note" },
          { type: "text", text: "User interruption after dropped turn" },
        ],
      },
    ]

    const plan = resolveCheckpointRewind({
      cachedSession: {
        claudeSessionId: "sdk-1",
        lastAccess: Date.now(),
        messageCount: 3,
        lineageHash: "h",
        sdkMessageUuids: [null, "asst-uuid-1", null, "dropped-asst-uuid-3"],
        passthroughToolCallAssistantUuid: "dropped-asst-uuid-3",
        passthroughToolCallIds: ["toolu_dropped"],
      },
      resumeFrom: 2,
      allMessages: messages,
      allowTrailingSystemReminder: true,
    })

    expect(plan).toBeDefined()
    expect(plan?.kind).toBe("tool_checkpoint")
    if (plan?.kind === "tool_checkpoint") {
      expect(plan.assistantUuid).toBe("asst-uuid-1")
      expect(plan.toolIds).toEqual(["toolu_a"])
      expect(plan.prefixLen).toBe(2)
      expect(plan.messagesToConvert).toHaveLength(1)
      expect(plan.messagesToConvert[0]!.content).toHaveLength(3)
    }
  })
})

describe("prefix-based fork seeding and checkpoint rewind (proxy integration)", () => {
  const post = async (
    app: { fetch: (request: Request) => Response | Promise<Response> },
    session: string,
    messages: Array<{ role: string; content: any }>,
  ) => {
    const response = await app.fetch(new Request("http://localhost/v1/messages", {
      method: "POST",
      headers: {
        "content-type": "application/json",
        "x-session-affinity": session,
      },
      body: JSON.stringify({ model: "claude-haiku-5-5", max_tokens: 128, stream: false, messages }),
    }))
    expect(response.status).toBe(200)
    await response.text()
    return captured.at(-1)!
  }

  it("forks a multi-turn conversation by prefix even when x-meridian-fork-from is stripped", async () => {
    const { app } = createProxyServer({ port: 0, host: "127.0.0.1", silent: true })
    const parent = crypto.randomUUID()
    const turn1 = [{ role: "user", content: "PREFIX_FORK_TURN_1" }]
    await post(app, parent, turn1)
    const turn2 = [...turn1, { role: "assistant", content: "acknowledged" }, { role: "user", content: "PREFIX_FORK_TURN_2" }]
    await post(app, parent, turn2)
    const source = lookupSharedSession(parent)!

    const lane = crypto.randomUUID()
    const laneCall = await post(app, lane, [
      ...turn2,
      { role: "assistant", content: "acknowledged" },
      { role: "user", content: "Lane brief without fork header." },
    ])

    expect(laneCall.options.resume).toBe(source.claudeSessionId)
    expect(laneCall.prompt).not.toContain("PREFIX_FORK_TURN_1")
    expect(laneCall.prompt).toContain("Lane brief without fork header.")
    expect(lookupSharedSession(parent)!.claudeSessionId).toBe(source.claudeSessionId)
  })

  it("preserves structured tool_result blocks when an undo fork rewinds to an assistant tool_use turn", async () => {
    const { app } = createProxyServer({ port: 0, host: "127.0.0.1", silent: true })
    const parent = crypto.randomUUID()
    const parentHistory = [
      { role: "user", content: "UNDO_TOOL_FORK_ORIGIN" },
      { role: "assistant", content: [{ type: "tool_use", id: "toolu_split_undo", name: "split", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_split_undo", content: "Parent started lanes" }] },
    ]
    await storeSharedSession(
      parent,
      "sdk-parent-tool-undo",
      parentHistory.length,
      computeLineageHash(parentHistory),
      computeMessageHashes(parentHistory),
      [null, "asst-split-uuid-1", null, "asst-after-split-uuid-3"],
    )

    const lane = crypto.randomUUID()
    const laneCall = await post(app, lane, [
      parentHistory[0]!,
      parentHistory[1]!,
      { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_split_undo", content: "Lane assignment payload" }] },
    ])

    expect(laneCall.options.resume).toBe("sdk-parent-tool-undo")
    expect(laneCall.options.resumeSessionAt).toBe("asst-split-uuid-1")
    expect(laneCall.prompt).not.toContain("UNDO_TOOL_FORK_ORIGIN")
    expect(laneCall.prompt).toContain("toolu_split_undo")
    expect(laneCall.prompt).toContain("Lane assignment payload")
  })

  it("rebinds a stale unsettled checkpoint to a sibling fallback key that holds more of the history", async () => {
    const { app } = createProxyServer({ port: 0, host: "127.0.0.1", silent: true })
    const baseKey = crypto.randomUUID()
    const fallbackKey = `${baseKey}:run-1:fallback:1`
    const prefix = [
      { role: "user", content: "FALLBACK_REBIND_TURN_1" },
      { role: "assistant", content: [{ type: "tool_use", id: "toolu_1", name: "exec", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_1", content: "ok-1" }] },
    ]
    const advanced = [
      ...prefix,
      { role: "assistant", content: [{ type: "tool_use", id: "toolu_2", name: "exec", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_2", content: "ok-2" }] },
    ]
    // Stale baseKey has an unsettled checkpoint at msg 3.
    await storeSharedSession(
      baseKey,
      "sdk-stale-base",
      prefix.length,
      computeLineageHash(prefix),
      computeMessageHashes(prefix),
      [null, "asst-1", null, "stale-dropped-3"],
      undefined,
      undefined,
      "stale-dropped-3",
      ["toolu_lost_in_race"],
    )
    // Fallback key advanced the same conversation to msg 5 with pending toolu_3.
    await storeSharedSession(
      fallbackKey,
      "sdk-warm-fallback",
      advanced.length,
      computeLineageHash(advanced),
      computeMessageHashes(advanced),
      [null, "asst-1", null, "asst-3", null, "asst-5"],
      undefined,
      undefined,
      "asst-5",
      ["toolu_3"],
    )

    const nextCall = await post(app, baseKey, [
      ...advanced,
      { role: "assistant", content: [{ type: "tool_use", id: "toolu_3", name: "exec", input: {} }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_3", content: "ok-3" }] },
    ])

    expect(nextCall.options.resume).toBe("sdk-warm-fallback")
    expect(nextCall.options.resumeSessionAt).toBe("asst-5")
    expect(nextCall.prompt).not.toContain("FALLBACK_REBIND_TURN_1")
    expect(nextCall.prompt).toContain("ok-3")
  })

  it("rolls back to the intact prefix on middle context_replace excision and flattens preserved trailing turns", async () => {
    const { app } = createProxyServer({ port: 0, host: "127.0.0.1", silent: true })
    const key = crypto.randomUUID()
    const prefix = [
      { role: "user", content: "CONTEXT_REPLACE_ORIGIN" },
      { role: "assistant", content: [{ type: "tool_use", id: "toolu_15", name: "exec", input: { step: 15 } }] },
    ]
    const excisedMiddle = [
      { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_15", content: "result-15" }] },
      { role: "assistant", content: [{ type: "tool_use", id: "toolu_17", name: "exec", input: { step: 17 } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_17", content: "huge-output-17" }] },
      { role: "assistant", content: [{ type: "tool_use", id: "toolu_18", name: "exec", input: { step: 18 } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_18", content: "huge-output-18" }] },
    ]
    const preservedTail = [
      { role: "assistant", content: [{ type: "tool_use", id: "toolu_71", name: "exec", input: { step: 71 } }] },
      { role: "user", content: [{ type: "tool_result", tool_use_id: "toolu_71", content: "result-71" }] },
    ]
    const original = [...prefix, ...excisedMiddle, ...preservedTail]
    const uuids = original.map((m, i) => m.role === "assistant" ? `uuid-${i}` : null)

    // Verify lineage classifies middle excision with intact suffix as undo at prefixOverlap=2.
    const state: SessionState = {
      claudeSessionId: "sdk-context-replace",
      lastAccess: Date.now(),
      messageCount: original.length,
      lineageHash: computeLineageHash(original),
      messageHashes: computeMessageHashes(original),
      sdkMessageUuids: uuids,
    }
    const edited = [
      ...prefix,
      {
        role: "user",
        content: [
          { type: "tool_result", tool_use_id: "toolu_15", content: "result-15" },
          { type: "text", text: "{\"context_edit\":{\"from_call_number\":17,\"to_call_number\":18,\"output_summary\":\"summary\"}}" },
        ],
      },
      ...preservedTail,
    ]
    expect(verifyLineage(state, edited)).toMatchObject({
      type: "undo",
      prefixOverlap: 2,
      rollbackUuid: "uuid-1",
    })

    // Verify helper keeps only toolu_15 as structured tool_result and flattens toolu_71.
    const coalesced = buildToolUndoRewindMessages({
      tail: edited.slice(2),
      expectedToolIds: ["toolu_15"],
      allowTrailingSystemReminder: false,
    })
    expect(coalesced).toHaveLength(1)
    const blocks = coalesced![0]!.content as Array<{ type?: string; tool_use_id?: string; text?: string }>
    expect(blocks.filter(b => b.type === "tool_result").map(b => b.tool_use_id)).toEqual(["toolu_15"])
    expect(blocks.some(b => b.type === "text" && b.text?.includes("result-71"))).toBe(true)

    // Verify full proxy HTTP request forks at uuid-1 and never replays CONTEXT_REPLACE_ORIGIN or raw toolu_71 tool_result.
    await storeSharedSession(
      key,
      "sdk-context-replace",
      original.length,
      computeLineageHash(original),
      computeMessageHashes(original),
      uuids,
    )
    const call = await post(app, key, edited)
    expect(call.options.resume).toBe("sdk-context-replace")
    expect(call.options.resumeSessionAt).toBe("uuid-1")
    expect(call.prompt).not.toContain("CONTEXT_REPLACE_ORIGIN")
    expect(call.prompt).toContain("context_edit")
    expect(call.prompt).toContain("result-71")
    const rawContent = ((call.rawMessages[0] as { message?: { content?: unknown[] } })?.message?.content ?? []) as Array<{ type?: string; tool_use_id?: string }>
    expect(rawContent.filter(b => b.type === "tool_result").map(b => b.tool_use_id)).toEqual(["toolu_15"])
  })
})
