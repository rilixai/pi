// Runs one coding-agent task with the workspace's normal Pi resources.
import { stat } from "node:fs/promises";
import { isAbsolute } from "node:path";
import type { AgentMessage } from "@earendil-works/pi-agent-core";
import { getAgentDir } from "../config.ts";
import type { SessionStats } from "../core/agent-session.ts";
import { emitSessionShutdownEvent } from "../core/extensions/runner.ts";
import type { ModelRuntime } from "../core/model-runtime.ts";
import { DefaultResourceLoader } from "../core/resource-loader.ts";
import { createAgentSession } from "../core/sdk.ts";
import { SettingsManager } from "../core/settings-manager.ts";

export interface LocalRunRequest {
	prompt: string;
	cwd: string;
	model: string;
	skillPaths?: string[];
	appendSystemPrompt?: string;
}

export interface LocalRunResult {
	response: string;
	messages: AgentMessage[];
	stats: SessionStats;
	provider: "openai";
	model: string;
	cwd: string;
}

/** Validate the JSON request received from the Python subprocess boundary. */
export function parseLocalRunRequest(input: unknown): LocalRunRequest {
	if (!input || typeof input !== "object" || Array.isArray(input)) {
		throw new TypeError("Expected a JSON object containing prompt, cwd, and model.");
	}
	const value = input as Record<string, unknown>;
	for (const field of ["prompt", "cwd", "model"] as const) {
		if (typeof value[field] !== "string" || !value[field].trim()) {
			throw new TypeError(`${field} must be a nonempty string.`);
		}
	}
	if (
		value.skillPaths !== undefined &&
		(!Array.isArray(value.skillPaths) ||
			!value.skillPaths.every((path: unknown) => typeof path === "string" && path.trim().length > 0))
	) {
		throw new TypeError("skillPaths must be an array of nonempty paths.");
	}
	if (value.appendSystemPrompt !== undefined && typeof value.appendSystemPrompt !== "string") {
		throw new TypeError("appendSystemPrompt must be a string.");
	}
	// The fields crossing the process boundary have been checked above.
	const request = value as unknown as LocalRunRequest;
	if (!isAbsolute(request.cwd)) throw new TypeError("cwd must be an absolute workspace path.");
	return request;
}

/** Run a fresh session to completion without copying or deleting the workspace. */
export async function runLocalAgent(
	request: LocalRunRequest,
	modelRuntime: ModelRuntime,
	signal: AbortSignal,
): Promise<LocalRunResult> {
	signal.throwIfAborted();
	if (!(await stat(request.cwd)).isDirectory()) throw new TypeError("cwd must be an existing directory.");
	const model = modelRuntime.getModel("openai", request.model);
	if (!model) throw new Error(`OpenAI model not found in Pi's catalog: ${request.model}`);
	const agentDir = getAgentDir();
	const settingsManager = SettingsManager.create(request.cwd, agentDir);
	const appendSystemPrompt = request.appendSystemPrompt;
	const resourceLoader = new DefaultResourceLoader({
		cwd: request.cwd,
		agentDir,
		settingsManager,
		additionalSkillPaths: request.skillPaths,
		appendSystemPromptOverride:
			appendSystemPrompt === undefined ? undefined : (base) => [...base, appendSystemPrompt],
	});
	await resourceLoader.reload();
	signal.throwIfAborted();
	const { session } = await createAgentSession({
		cwd: request.cwd,
		agentDir,
		model,
		modelRuntime,
		settingsManager,
		resourceLoader,
	});
	let abortPromise: Promise<void> | undefined;
	const abort = () => {
		abortPromise ??= session.abort();
	};
	signal.addEventListener("abort", abort, { once: true });
	try {
		signal.throwIfAborted();
		await session.bindExtensions({
			mode: "print",
			onError: (error) => console.error(`Extension error (${error.extensionPath}): ${error.error}`),
		});
		signal.throwIfAborted();
		await session.prompt(request.prompt);
		await session.waitForIdle();
		signal.throwIfAborted();
		const lastAssistant = session.messages.findLast((message) => message.role === "assistant");
		if (!lastAssistant) throw new Error("Pi finished without an assistant response.");
		if (lastAssistant.stopReason !== "stop") {
			throw new Error(lastAssistant.errorMessage ?? `Pi stopped with reason: ${lastAssistant.stopReason}`);
		}
		return {
			response: session.getLastAssistantText() ?? "",
			messages: session.messages,
			stats: session.getSessionStats(),
			provider: "openai",
			model: model.id,
			cwd: request.cwd,
		};
	} finally {
		signal.removeEventListener("abort", abort);
		try {
			await abortPromise;
			await emitSessionShutdownEvent(session.extensionRunner, { type: "session_shutdown", reason: "quit" });
		} finally {
			session.dispose();
		}
	}
}
