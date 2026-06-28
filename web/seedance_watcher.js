import { app } from "../../scripts/app.js";
import { api } from "../../scripts/api.js";

// Toasts for the non-blocking Seedance watcher. The Python watcher (running on
// ComfyUI's event loop, NOT a prompt execution) pushes these events when a
// fire-and-forget job finishes, so the user is notified without polling.
app.registerExtension({
    name: "zerogen.seedance.watcher",
    setup() {
        const notify = (severity, summary, detail, life = 7000) => {
            try {
                app.extensionManager?.toast?.add({ severity, summary, detail, life });
            } catch (e) {
                /* extensionManager not ready / unavailable — console still fires */
            }
            console.log(`[Seedance] ${summary} — ${detail ?? ""}`);
        };

        api.addEventListener("zerogen.seedance.done", (e) => {
            const d = e.detail || {};
            notify(
                "success",
                `Seedance ready: ${d.label || d.task_id}`,
                `${d.elapsed_s != null ? d.elapsed_s + "s" : "done"} → ${d.saved_video || ""}`,
                9000,
            );
        });

        api.addEventListener("zerogen.seedance.failed", (e) => {
            const d = e.detail || {};
            notify(
                "error",
                `Seedance failed: ${d.label || d.task_id}`,
                `${d.status || "error"}: ${d.error || ""} (recover by task_id with Seedance Fetch Task)`,
                12000,
            );
        });

        // Status transitions (submitted → queued/running) are console-only to
        // avoid toast spam; the heartbeat lives in the server console.
        api.addEventListener("zerogen.seedance.update", (e) => {
            const d = e.detail || {};
            console.log(`[Seedance] ${d.label || d.task_id}: ${d.status}`);
        });
    },
});
