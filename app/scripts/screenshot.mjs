// Headless Chromium screenshots over the DevTools protocol. No dependencies
// beyond Node 24 (built-in WebSocket) and a `chromium-browser` binary.
//
//   pnpm screenshot <url> <out.png> [--dark|--light] [--width 430] [--height 932]
//        [--wait ms] [--click "css"]... [--eval "js"]... [--file "css" "/path"]... [--sleep ms]...
//
// Steps (--click / --eval / --file / --sleep) run in the order given, each
// followed by a short settle delay, then the screenshot is taken. A fresh
// browser profile is used every run so the service worker never serves a
// previous build. Console errors and uncaught exceptions are printed.
//
// Example: open the scanner, upload a photo, capture the review screen.
//   pnpm screenshot http://127.0.0.1:3000/ review.png --dark \
//     --click 'button[aria-label="Scan a card"]' --sleep 3000 \
//     --file 'input[type=file]' /path/to/photo.jpg --sleep 12000
import { spawn } from 'node:child_process';
import { mkdtempSync, rmSync, writeFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const args = process.argv.slice(2);
const url = args.shift();
const out = args.shift();
if (!url || !out) {
	console.error('usage: screenshot.mjs <url> <out.png> [options]');
	process.exit(2);
}
const opts = { dark: true, width: 430, height: 932, wait: 800, steps: [] };
for (let i = 0; i < args.length; i++) {
	const a = args[i];
	if (a === '--dark') opts.dark = true;
	else if (a === '--light') opts.dark = false;
	else if (a === '--width') opts.width = Number(args[++i]);
	else if (a === '--height') opts.height = Number(args[++i]);
	else if (a === '--wait') opts.wait = Number(args[++i]);
	else if (a === '--click') opts.steps.push({ click: args[++i] });
	else if (a === '--eval') opts.steps.push({ eval: args[++i] });
	else if (a === '--file') opts.steps.push({ file: args[++i], path: args[++i] });
	else if (a === '--sleep') opts.steps.push({ sleep: Number(args[++i]) });
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));
const port = 9222 + Math.floor(Math.random() * 500);
const profile = mkdtempSync(join(tmpdir(), 'screenshot-profile-'));
const chrome = spawn(
	process.env.CHROMIUM ?? 'chromium-browser',
	[
		'--headless=new',
		`--remote-debugging-port=${port}`,
		'--no-first-run',
		'--no-default-browser-check',
		'--disable-gpu',
		'--hide-scrollbars',
		`--window-size=${opts.width},${opts.height}`,
		`--user-data-dir=${profile}`,
		'about:blank',
	],
	{ stdio: ['ignore', 'ignore', 'pipe'] },
);

let wsUrl;
for (let i = 0; i < 50 && !wsUrl; i++) {
	try {
		const list = await (await fetch(`http://127.0.0.1:${port}/json`)).json();
		wsUrl = list.find((t) => t.type === 'page')?.webSocketDebuggerUrl;
	} catch {}
	if (!wsUrl) await sleep(100);
}
if (!wsUrl) {
	chrome.kill();
	throw new Error('Chromium did not start (set CHROMIUM to the binary path)');
}

const ws = new WebSocket(wsUrl);
await new Promise((r) => {
	ws.onopen = r;
});
let id = 0;
const pending = new Map();
const events = [];
ws.onmessage = (m) => {
	const msg = JSON.parse(m.data);
	if (msg.id && pending.has(msg.id)) {
		pending.get(msg.id)(msg);
		pending.delete(msg.id);
	} else if (msg.method) events.push(msg);
};
const send = (method, params = {}) =>
	new Promise((resolve, reject) => {
		const i = ++id;
		pending.set(i, (msg) =>
			msg.error ? reject(new Error(`${method}: ${msg.error.message}`)) : resolve(msg.result),
		);
		ws.send(JSON.stringify({ id: i, method, params }));
	});
const evaluate = async (expression) => {
	const r = await send('Runtime.evaluate', { expression, awaitPromise: true, returnByValue: true });
	if (r.exceptionDetails) {
		throw new Error(`eval failed: ${r.exceptionDetails.exception?.description ?? r.exceptionDetails.text}`);
	}
	return r.result.value;
};

await send('Page.enable');
await send('Runtime.enable');
await send('DOM.enable');
await send('Emulation.setDeviceMetricsOverride', {
	width: opts.width,
	height: opts.height,
	deviceScaleFactor: 2,
	mobile: opts.width < 700,
});
await send('Emulation.setEmulatedMedia', {
	features: [{ name: 'prefers-color-scheme', value: opts.dark ? 'dark' : 'light' }],
});
await send('Page.navigate', { url });
await sleep(opts.wait);

for (const step of opts.steps) {
	if (step.click) {
		const ok = await evaluate(
			`(() => { const el = document.querySelector(${JSON.stringify(step.click)}); if (!el) return false; el.click(); return true; })()`,
		);
		if (!ok) console.error(`click: no element for ${step.click}`);
	} else if (step.eval) {
		await evaluate(step.eval);
	} else if (step.file) {
		const { root } = await send('DOM.getDocument');
		const { nodeId } = await send('DOM.querySelector', { nodeId: root.nodeId, selector: step.file });
		await send('DOM.setFileInputFiles', { nodeId, files: [step.path] });
	} else if (step.sleep) {
		await sleep(step.sleep);
		continue;
	}
	await sleep(500);
}

const { data } = await send('Page.captureScreenshot', { format: 'png' });
writeFileSync(out, Buffer.from(data, 'base64'));
for (const ev of events) {
	if (
		ev.method === 'Runtime.consoleAPICalled' &&
		(ev.params.type === 'error' || ev.params.type === 'warning')
	) {
		console.error(
			`console.${ev.params.type}:`,
			ev.params.args.map((a) => a.value ?? a.description).join(' '),
		);
	}
	if (ev.method === 'Runtime.exceptionThrown') {
		console.error(
			'exception:',
			ev.params.exceptionDetails.exception?.description ?? ev.params.exceptionDetails.text,
		);
	}
}
ws.close();
chrome.kill();
await sleep(300);
rmSync(profile, { recursive: true, force: true });
console.log(`wrote ${out}`);
