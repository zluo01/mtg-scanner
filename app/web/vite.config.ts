import tailwindcss from '@tailwindcss/vite';
import { defineConfig } from 'vite';
import solid from 'vite-plugin-solid';

// In development the API server runs separately (server/, port 3000) and
// Vite proxies the dynamic routes to it. In production the server serves
// this build itself, so every path below is same-origin and relative.
const API = process.env.API_URL ?? 'http://127.0.0.1:3000';

export default defineConfig({
	plugins: [solid(), tailwindcss()],
	resolve: { tsconfigPaths: true },
	server: {
		// Listen on every interface so a phone on the LAN can use the dev server too.
		host: true,
		port: 5173,
		proxy: {
			'/api': API,
			'/scans': API,
			'/models': API,
		},
	},
	build: {
		// One level up so the server (app/server) and the output (app/dist) sit side by side.
		outDir: '../dist',
		emptyOutDir: true,
		target: 'es2022',
		// onnxruntime-web is large but only loaded when the scanner opens.
		chunkSizeWarningLimit: 1500,
	},
});
