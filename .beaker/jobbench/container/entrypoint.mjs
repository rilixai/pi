// Bridge native Pi HTTP requests to the per-task Unix socket without network access.
import { spawn } from "node:child_process";
import { mkdirSync, writeFileSync } from "node:fs";
import http from "node:http";

const agentDir = process.env.PI_CODING_AGENT_DIR;
mkdirSync(agentDir, { recursive: true });
writeFileSync(`${agentDir}/settings.json`, JSON.stringify({ defaultThinkingLevel: process.env.PI_THINKING_LEVEL }));
writeFileSync(`${agentDir}/models.json`, JSON.stringify({ providers: { openai: { baseUrl: "http://127.0.0.1:3000/v1" } } }));

const server = http.createServer((request, response) => {
	const upstream = http.request({
		socketPath: "/run/pi-bridge/openai.sock",
		method: request.method,
		path: request.url,
		headers: request.headers,
	}, (incoming) => {
		response.writeHead(incoming.statusCode, incoming.headers);
		incoming.pipe(response);
	});
	upstream.on("error", () => {
		if (!response.headersSent) response.writeHead(502);
		response.end("Pi model relay unavailable");
	});
	response.on("close", () => upstream.destroy());
	request.pipe(upstream);
});
await new Promise((resolve, reject) => {
	server.once("error", reject);
	server.listen(3000, "127.0.0.1", resolve);
});
const directory = "/opt/pi/packages/coding-agent/src/experimental";
const child = spawn(process.execPath, ["--import", `file://${directory}/source-resolver.ts`, `${directory}/local-runner-cli.ts`], {
	stdio: "inherit",
});
for (const signal of ["SIGTERM", "SIGINT"]) process.on(signal, () => child.kill(signal));
child.on("error", (error) => {
	console.error(error.message);
	process.exit(1);
});
child.on("exit", (code) => {
	server.closeAllConnections();
	server.close();
	process.exit(code ?? 1);
});
