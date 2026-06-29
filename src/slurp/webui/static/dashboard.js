/* slurp dashboard — vanilla JS, no build step */

(function () {
    'use strict';

    // ── Config ──
    const TOKEN = document.querySelector('meta[name="stream-token"]')?.content || '';
    const API_BASE = '';
    const MAX_RECONNECT_DELAY = 30000;

    // ── State ──
    const jobs = new Map();
    let selectedJobId = null;
    let csrfToken = null;
    let reconnectDelay = 1000;
    let reconnectTimer = null;
    let sse = null;
    let logFollower = null;
    let experimentDebounce = null;
    let profiles = new Set();

    // ── Status priority ──
    const STATUS_PRIORITY = {
        RUNNING: 0,
        PENDING: 1,
        COMPLETED: 2,
        FAILED: 3,
        TIMEOUT: 4,
        CANCELLED: 5,
        UNKNOWN: 6,
    };

    const STATUS_CLASS = {
        PENDING: 'badge-pending',
        RUNNING: 'badge-running',
        COMPLETED: 'badge-completed',
        FAILED: 'badge-failed',
        CANCELLED: 'badge-cancelled',
        TIMEOUT: 'badge-timeout',
        UNKNOWN: 'badge-cancelled',
    };

    // ── DOM refs ──
    const $ = (sel) => document.querySelector(sel);
    const $$ = (sel) => document.querySelectorAll(sel);
    const tbody = $('#jobs-tbody');
    const cardList = document.createElement('div');
    cardList.className = 'card-list';
    $('.table-wrap').appendChild(cardList);
    const detailPanel = $('#detail-panel');
    const detailContent = $('#detail-content');
    const logPanel = $('#log-panel');
    const logOutput = $('#log-output');
    const logTitle = $('#log-title');
    const toastContainer = $('#toast-container');
    const experimentInput = $('#experiment-filter');
    const statusFilter = $('#status-filter');
    const partitionFilter = $('#partition-filter');
    const profileSelect = $('#profile-selector');
    const jobCount = $('#job-count');

    // ── Utilities ──
    function apiUrl(path) {
        const sep = path.includes('?') ? '&' : '?';
        return `${API_BASE}${path}${sep}token=${encodeURIComponent(TOKEN)}`;
    }

    async function fetchCsrf() {
        try {
            const res = await fetch(apiUrl('/api/csrf-token'));
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            csrfToken = data.csrf_token;
        } catch (e) {
            showToast('Failed to fetch CSRF token: ' + e.message, 'error');
        }
    }

    function showToast(msg, type = 'info') {
        const el = document.createElement('div');
        el.className = `toast toast-${type}`;
        el.textContent = msg;
        toastContainer.appendChild(el);
        setTimeout(() => {
            el.classList.add('toast-out');
            el.addEventListener('animationend', () => el.remove());
        }, 4000);
    }

    function escapeHtml(str) {
        if (!str) return '';
        return str.replace(/[&<>"']/g, (m) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[m]);
    }

    function formatTime(str) {
        if (!str) return '—';
        // [[HH:]MM:]SS or D-HH:MM:SS
        return str;
    }

    function progressForStatus(status) {
        if (status === 'RUNNING') return 50;
        if (status === 'COMPLETED') return 100;
        if (status === 'PENDING') return 10;
        return 0;
    }

    function etaForJob(job) {
        const res = job.resources || {};
        const elapsed = job.elapsed || '0';
        const timeLimit = res.time || '2:00:00';
        // Very rough heuristic
        return '—';
    }

    // ── SSE ──
    function connectSSE() {
        if (sse) { sse.close(); sse = null; }
        const url = apiUrl('/stream');
        sse = new EventSource(url);

        sse.addEventListener('job_update', (e) => {
            try {
                const job = JSON.parse(e.data);
                jobs.set(job.job_id, job);
                if (job.profile) profiles.add(job.profile);
                renderJobs();
                if (selectedJobId === job.job_id) renderDetail(job);
            } catch (err) {
                console.error('job_update parse error', err);
            }
        });

        sse.addEventListener('log_append', (e) => {
            try {
                const data = JSON.parse(e.data);
                if (data.job_id === selectedJobId && data.text) {
                    appendLog(data.text);
                }
            } catch (err) {
                console.error('log_append parse error', err);
            }
        });

        sse.addEventListener('heartbeat', () => {
            reconnectDelay = 1000;
        });

        sse.addEventListener('server_error', (e) => {
            showToast('Server error: ' + (e.data || 'unknown'), 'error');
        });

        sse.onerror = () => {
            sse.close();
            sse = null;
            showToast('Connection lost. Reconnecting…', 'error');
            scheduleReconnect();
        };

        sse.onopen = () => {
            reconnectDelay = 1000;
            showToast('Connected', 'success');
        };
    }

    function scheduleReconnect() {
        if (reconnectTimer) return;
        reconnectTimer = setTimeout(() => {
            reconnectTimer = null;
            connectSSE();
        }, reconnectDelay);
        reconnectDelay = Math.min(reconnectDelay * 2, MAX_RECONNECT_DELAY);
    }

    // ── Job list ──
    function sortJobs(jobArray) {
        return jobArray.sort((a, b) => {
            const pa = STATUS_PRIORITY[a.status] ?? 99;
            const pb = STATUS_PRIORITY[b.status] ?? 99;
            if (pa !== pb) return pa - pb;
            return (a.job_id || '').localeCompare(b.job_id || '');
        });
    }

    function filterJobs() {
        const exp = experimentInput.value.trim().toLowerCase();
        const status = statusFilter.value;
        const part = partitionFilter.value.trim().toLowerCase();
        const prof = profileSelect.value;
        return Array.from(jobs.values()).filter((j) => {
            if (exp && !(j.experiment || '').toLowerCase().includes(exp)) return false;
            if (status && j.status !== status) return false;
            if (part && !(j.resources?.partition || '').toLowerCase().includes(part)) return false;
            if (prof && j.profile !== prof) return false;
            return true;
        });
    }

    function renderJobs() {
        const filtered = sortJobs(filterJobs());
        jobCount.textContent = `${filtered.length} job${filtered.length === 1 ? '' : 's'}`;
        updateProfileOptions();

        if (filtered.length === 0) {
            tbody.innerHTML = '<tr class="empty-row"><td colspan="9">No jobs match.</td></tr>';
            cardList.innerHTML = '<div class="empty-row">No jobs match.</div>';
            return;
        }

        const rows = filtered.map((j) => {
            const res = j.resources || {};
            const progress = progressForStatus(j.status);
            const badgeClass = STATUS_CLASS[j.status] || 'badge-cancelled';
            return `
                <tr data-id="${escapeHtml(j.job_id)}" class="${j.job_id === selectedJobId ? 'selected' : ''}">
                    <td>${escapeHtml(j.job_id)}</td>
                    <td>${escapeHtml(j.name || '—')}</td>
                    <td><span class="badge ${badgeClass}">${escapeHtml(j.status)}</span></td>
                    <td>${escapeHtml(res.partition || '—')}</td>
                    <td>${escapeHtml(String(res.nodes || 1))}</td>
                    <td>${escapeHtml(String(res.gpus || 0))}</td>
                    <td>${escapeHtml(formatTime(res.time))}</td>
                    <td>
                        <div class="progress-bar"><div class="progress-fill" style="width:${progress}%"></div></div>
                    </td>
                    <td>${escapeHtml(etaForJob(j))}</td>
                </tr>
            `;
        }).join('');
        tbody.innerHTML = rows;

        const cards = filtered.map((j) => {
            const res = j.resources || {};
            const badgeClass = STATUS_CLASS[j.status] || 'badge-cancelled';
            return `
                <div class="job-card" data-id="${escapeHtml(j.job_id)}">
                    <div class="job-card-header">
                        <span class="job-card-title">${escapeHtml(j.name || j.job_id)}</span>
                        <span class="badge ${badgeClass}">${escapeHtml(j.status)}</span>
                    </div>
                    <div class="job-card-meta">
                        <span>ID: ${escapeHtml(j.job_id)}</span>
                        <span>Partition: ${escapeHtml(res.partition || '—')}</span>
                        <span>Nodes: ${escapeHtml(String(res.nodes || 1))}</span>
                        <span>GPUs: ${escapeHtml(String(res.gpus || 0))}</span>
                    </div>
                </div>
            `;
        }).join('');
        cardList.innerHTML = cards;
    }

    function updateProfileOptions() {
        const current = profileSelect.value;
        const existing = new Set();
        for (const opt of profileSelect.querySelectorAll('option')) {
            if (opt.value) existing.add(opt.value);
        }
        let changed = false;
        for (const p of profiles) {
            if (!existing.has(p)) {
                const opt = document.createElement('option');
                opt.value = p;
                opt.textContent = p;
                profileSelect.appendChild(opt);
                changed = true;
            }
        }
        if (changed && current) profileSelect.value = current;
    }

    // ── Detail panel ──
    function renderDetail(job) {
        const res = job.resources || {};
        const metrics = job.metrics || {};
        detailContent.innerHTML = `
            <div class="detail-section">
                <h3>Job</h3>
                <p><strong>${escapeHtml(job.name || '—')}</strong> <code>${escapeHtml(job.job_id)}</code></p>
            </div>
            <div class="detail-section">
                <h3>Status</h3>
                <p><span class="badge ${STATUS_CLASS[job.status] || 'badge-cancelled'}">${escapeHtml(job.status)}</span></p>
            </div>
            <div class="detail-section">
                <h3>Profile</h3>
                <p>${escapeHtml(job.profile || '—')}</p>
            </div>
            <div class="detail-section">
                <h3>Experiment</h3>
                <p>${escapeHtml(job.experiment || '—')}</p>
            </div>
            <div class="detail-section">
                <h3>Command</h3>
                <pre>${escapeHtml(job.command || '—')}</pre>
            </div>
            <div class="detail-section">
                <h3>Resources</h3>
                <p>
                    Partition: ${escapeHtml(res.partition || '—')}<br>
                    Nodes: ${escapeHtml(String(res.nodes || 1))}<br>
                    GPUs: ${escapeHtml(String(res.gpus || 0))}<br>
                    CPUs: ${escapeHtml(String(res.cpus || 8))}<br>
                    Time: ${escapeHtml(formatTime(res.time))}<br>
                    Memory: ${escapeHtml(res.mem || '—')}<br>
                    Account: ${escapeHtml(res.account || '—')}
                </p>
            </div>
            <div class="detail-section">
                <h3>Working Directory</h3>
                <p>${escapeHtml(job.working_dir || '—')}</p>
            </div>
            ${Object.keys(metrics).length ? `
            <div class="detail-section">
                <h3>Metrics</h3>
                <pre>${escapeHtml(JSON.stringify(metrics, null, 2))}</pre>
            </div>
            ` : ''}
            <div class="detail-section">
                <button id="cancel-btn" class="btn btn-danger" ${['RUNNING','PENDING'].includes(job.status) ? '' : 'disabled'}>Cancel Job</button>
                <button id="logs-btn" class="btn btn-ghost">View Logs</button>
            </div>
        `;
        detailPanel.classList.add('open');

        $('#cancel-btn')?.addEventListener('click', () => cancelJob(job.job_id));
        $('#logs-btn')?.addEventListener('click', () => openLogs(job.job_id));
    }

    async function selectJob(jobId) {
        selectedJobId = jobId;
        const job = jobs.get(jobId);
        if (job) {
            renderDetail(job);
            // Also fetch fresh logs preview
            await fetchLogPreview(jobId);
        }
        renderJobs();
    }

    async function fetchLogPreview(jobId) {
        try {
            const res = await fetch(apiUrl(`/api/jobs/${encodeURIComponent(jobId)}/logs?follow=false`));
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const text = await res.text();
            const lines = text.split('\n').slice(-50);
            logOutput.textContent = lines.join('\n');
            logTitle.textContent = `Logs — ${jobId}`;
        } catch (e) {
            logOutput.textContent = `Error loading logs: ${e.message}`;
        }
    }

    function appendLog(text) {
        if (!text) return;
        const txt = logOutput.textContent;
        logOutput.textContent = (txt ? txt + '\n' : '') + text;
        logOutput.scrollTop = logOutput.scrollHeight;
    }

    function openLogs(jobId) {
        logPanel.classList.add('open');
        logTitle.textContent = `Logs — ${jobId}`;
        fetchLogPreview(jobId);
    }

    // ── Actions ──
    async function cancelJob(jobId) {
        if (!csrfToken) await fetchCsrf();
        try {
            const res = await fetch(apiUrl(`/api/jobs/${encodeURIComponent(jobId)}/cancel`), {
                method: 'POST',
                headers: { 'X-CSRF-Token': csrfToken || '' },
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            showToast(data.message || 'Cancel requested', 'success');
            if (data.job) jobs.set(data.job.job_id, data.job);
            renderJobs();
        } catch (e) {
            showToast('Cancel failed: ' + e.message, 'error');
        }
    }

    async function syncCode() {
        if (!csrfToken) await fetchCsrf();
        try {
            const res = await fetch(apiUrl('/api/sync'), {
                method: 'POST',
                headers: { 'X-CSRF-Token': csrfToken || '' },
            });
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            showToast(data.message || 'Sync complete', 'success');
        } catch (e) {
            showToast('Sync failed: ' + e.message, 'error');
        }
    }

    async function refreshJobs() {
        try {
            const res = await fetch(apiUrl(`/api/jobs?experiment=${encodeURIComponent(experimentInput.value.trim())}`));
            if (!res.ok) throw new Error(`HTTP ${res.status}`);
            const data = await res.json();
            jobs.clear();
            for (const j of data) {
                jobs.set(j.job_id, j);
                if (j.profile) profiles.add(j.profile);
            }
            renderJobs();
            showToast('Jobs refreshed', 'success');
        } catch (e) {
            showToast('Refresh failed: ' + e.message, 'error');
        }
    }

    // ── Event listeners ──
    tbody.addEventListener('click', (e) => {
        const row = e.target.closest('tr[data-id]');
        if (row) selectJob(row.dataset.id);
    });

    cardList.addEventListener('click', (e) => {
        const card = e.target.closest('.job-card[data-id]');
        if (card) selectJob(card.dataset.id);
    });

    $('#refresh-btn').addEventListener('click', refreshJobs);
    $('#sync-btn').addEventListener('click', syncCode);
    $('#detail-close').addEventListener('click', () => {
        detailPanel.classList.remove('open');
        selectedJobId = null;
        renderJobs();
    });
    $('#log-close').addEventListener('click', () => logPanel.classList.remove('open'));
    $('#log-toggle').addEventListener('click', () => logPanel.classList.toggle('open'));
    $('#sidebar-toggle').addEventListener('click', () => {
        const sb = $('#sidebar');
        sb.classList.toggle('open');
    });
    $('#sidebar-close').addEventListener('click', () => $('#sidebar').classList.remove('open'));

    experimentInput.addEventListener('input', () => {
        clearTimeout(experimentDebounce);
        experimentDebounce = setTimeout(() => {
            refreshJobs();
        }, 300);
    });

    statusFilter.addEventListener('change', renderJobs);
    partitionFilter.addEventListener('input', () => {
        clearTimeout(experimentDebounce);
        experimentDebounce = setTimeout(renderJobs, 300);
    });
    profileSelect.addEventListener('change', renderJobs);

    // ── Init ──
    async function init() {
        await fetchCsrf();
        await refreshJobs();
        connectSSE();
    }

    init();
})();
