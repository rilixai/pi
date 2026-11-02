// JSON subprocess entry point for scripts/pi_harness.py. Preload source-resolver.ts.
import { configureHttpDispatcher } from "../core/http-dispatcher.ts";
import { ModelRuntime } from "../core/model-runtime.ts";
import { flushRawStdout, takeOverStdout, writeRawStdout } from "../core/output-guard.ts";
import { killTrackedDetachedChildren } from "../utils/shell.ts";
import { parseLocalRunRequest, runLocalAgent } from "./local-runner.ts";

takeOverStdout();
const controller = new AbortController();
const abort = () => {
	// Pi's bash tools create detached process groups, outside Python's Node group.
	killTrackedDetachedChildren();
	controller.abort(new Error("Pi run interrupted."));
};
process.on("SIGTERM", abort);
process.on("SIGINT", abort);

let exitCode = 0;
try {
	let input = "";
	process.stdin.setEncoding("utf8");
	for await (const chunk of process.stdin) input += chunk;
	const request = parseLocalRunRequest(JSON.parse(input));
	process.chdir(request.cwd);
	const apiKey = process.env.OPENAI_API_KEY;
	if (!apiKey?.trim()) throw new Error("OPENAI_API_KEY is required for local OpenAI runs.");
	configureHttpDispatcher();
	const modelRuntime = await ModelRuntime.create({ refreshOnCreate: false, signal: controller.signal });
	// Select API-key authentication even if the user's Pi config has a subscription login.
	await modelRuntime.setRuntimeApiKey("openai", apiKey, { signal: controller.signal });
	const result = await runLocalAgent(request, modelRuntime, controller.signal);
	writeRawStdout(`${JSON.stringify({ ok: true, result })}\n`);
} catch (error) {
	exitCode = 1;
	const message = error instanceof Error ? error.message : String(error);
	console.error(message);
	writeRawStdout(
		`${JSON.stringify({ ok: false, error: { name: error instanceof Error ? error.name : "Error", message } })}\n`,
	);
} finally {
	killTrackedDetachedChildren();
	process.off("SIGTERM", abort);
	process.off("SIGINT", abort);
}
await flushRawStdout();
// Extensions may retain timers or connections after their shutdown hooks complete.
process.exit(exitCode);
