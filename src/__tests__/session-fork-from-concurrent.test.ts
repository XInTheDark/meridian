import { beforeEach, describe, expect, it } from "bun:test"
import type { Options } from "@anthropic-ai/claude-agent-sdk"
import { installSdkMock } from "./sdkMock"
import { installLoggerMock } from "./loggerMock"
import { installMcpToolsMock } from "./mcpToolsMock"
import { assistantMessage, withMockSdkSessionId } from "./helpers"

interface CapturedQuery {
  prompt: string | AsyncIterable<unknown>
  options: Options
}
let hold: Promise<void> | undefined
const captured: Array<{ options: Options; prompt: string }> = []
installSdkMock(() => ({
  query: (params: CapturedQuery) => (async function* () {
    let prompt = ""
    if (typeof params.prompt === "string") prompt = params.prompt
    else for await (const message of params.prompt) prompt += JSON.stringify(message)
    captured.push({ options: params.options, prompt })
    if (prompt.includes("HOLD") && hold) await hold
    yield withMockSdkSessionId(assistantMessage([{ type: "text", text: "acknowledged" }]), params.options)
  })(),
  createSdkMcpServer: () => ({ type: "sdk", name: "test", instance: { tool: () => {}, registerTool: () => ({}) } }),
  tool: () => ({}),
}), "session-fork-from-concurrent.test.ts")
installLoggerMock(() => ({ claudeLog: () => {}, withClaudeLogContext: (_ctx: unknown, fn: () => unknown) => fn() }))
installMcpToolsMock(() => ({ createOpencodeMcpServer: () => ({ type: "sdk", name: "opencode", instance: {} }) }))

const { createProxyServer, clearSessionCache } = await import("../proxy/server")
const { lookupSharedSession } = await import("../proxy/sessionStore")

beforeEach(async () => { await clearSessionCache(); captured.length = 0; hold = undefined })

type Message = { role: string; content: string }
type App = { fetch: (request: Request) => Response | Promise<Response> }

const send = async (app: App, session: string, messages: Message[], forkFrom?: string) => {
  const response = await app.fetch(new Request("http://localhost/v1/messages", {
    method: "POST",
    headers: {
      "content-type": "application/json",
      "x-session-affinity": session,
      ...(forkFrom ? { "x-meridian-fork-from": forkFrom } : {}),
    },
    body: JSON.stringify({ model: "claude-haiku-5-5", max_tokens: 128, stream: false, messages }),
  }))
  return { status: response.status, body: await response.text() }
}

describe("x-meridian-fork-from while another fork of the source runs", () => {
  it("lets a second fork and the source's own next turn run", async () => {
    const { app } = createProxyServer({ port: 0, host: "127.0.0.1", silent: true })
    const parent = crypto.randomUUID()
    const history: Message[] = [{ role: "user", content: "SHARED_HISTORY_CONCURRENT" }]
    expect((await send(app, parent, history)).status).toBe(200)
    const source = lookupSharedSession(parent)!.claudeSessionId
    const shared = [...history, { role: "assistant", content: "acknowledged" }]

    let release!: () => void
    hold = new Promise(resolve => { release = resolve })
    const first = send(app, crypto.randomUUID(), [...shared, { role: "user", content: "HOLD lane one." }], parent)
    while (!captured.some(call => call.prompt.includes("HOLD"))) await Bun.sleep(5)

    const second = await send(app, crypto.randomUUID(), [...shared, { role: "user", content: "Lane two." }], parent)
    expect(second.status).toBe(200)
    expect(captured.at(-1)!.options.resume).toBe(source)
    const own = await send(app, parent, [...shared, { role: "user", content: "Parent continues." }])
    expect(own.status).toBe(200)
    expect(captured.at(-1)!.options.resume).toBe(source)

    release()
    expect((await first).status).toBe(200)
  })
})
