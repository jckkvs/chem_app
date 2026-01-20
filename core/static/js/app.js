/**
 * ChemML Platform - Main JavaScript
 * モダンなSPA風UIのためのコアロジック
 */

// グローバル設定
const ChemML = {
    API_URL: '/api',
    REFRESH_INTERVAL: 10000, // 10秒
};

// モーダルダイアログ
class ModalDialog {
    constructor() {
        this.modal = null;
    }

    show(title, content) {
        const html = `
            <div class="modal-overlay" style="
                position: fixed; 
                top: 0; 
                left: 0; 
                width: 100%; 
                height: 100%; 
                background: rgba(0,0,0,0.7);
                display: flex;
                align-items: center;
                justify-content: center;
                z-index: 1000;
            ">
                <div class="modal-content" style="
                    background: var(--bg-card);
                    border-radius: 16px;
                    padding: 2rem;
                    max-width: 600px;
                    width: 90%;
                    max-height: 80vh;
                    overflow-y: auto;
                ">
                    <div style="display: flex; justify-content: space-between; align-items: center; margin-bottom: 1rem;">
                        <h2 style="color: var(--primary);">${title}</h2>
                        <button onclick="this.closest('.modal-overlay').remove()" class="btn btn-sm" style="background: var(--bg-hover);">✕</button>
                    </div>
                    <div>${content}</div>
                </div>
            </div>
        `;

        const div = document.createElement('div');
        div.innerHTML = html;
        document.body.appendChild(div.firstElementChild);
    }

    close() {
        const overlay = document.querySelector('.modal-overlay');
        if (overlay) overlay.remove();
    }
}

// データテーブル（ソート、フィルタ対応）
class DataTable {
    constructor(containerId, columns, fetchData) {
        this.container = document.getElementById(containerId);
        this.columns = columns;
        this.fetchData = fetchData;
        this.data = [];
        this.sortColumn = null;
        this.sortAscending = true;
        this.filterText = '';
    }

    async render() {
        this.data = await this.fetchData();

        let html = `
            <div style="margin-bottom: 1rem;">
                <input type="text" class="form-control" placeholder="🔍 検索..." 
                    onkeyup="this.closest('.card').dataTable.filter(this.value)"
                    style="max-width: 300px;">
            </div>
            <div style="overflow-x: auto;">
                <table class="table">
                    <thead>
                        <tr>
        `;

        this.columns.forEach((col, i) => {
            html += `<th style="cursor: pointer;" onclick="this.closest('.card').dataTable.sort(${i})">${col.label} ${this.sortColumn === i ? (this.sortAscending ? '↑' : '↓') : ''}</th>`;
        });

        html += `</tr></thead><tbody>`;

        const filteredData = this.filterText
            ? this.data.filter(row => JSON.stringify(row).toLowerCase().includes(this.filterText.toLowerCase()))
            : this.data;

        if (this.sortColumn !== null) {
            const col = this.columns[this.sortColumn];
            filteredData.sort((a, b) => {
                const aVal = col.getValue(a);
                const bVal = col.getValue(b);
                const result = aVal > bVal ? 1 : aVal < bVal ? -1 : 0;
                return this.sortAscending ? result : -result;
            });
        }

        filteredData.forEach(row => {
            html += '<tr>';
            this.columns.forEach(col => {
                html += `<td>${col.render ? col.render(row) : col.getValue(row)}</td>`;
            });
            html += '</tr>';
        });

        html += `</tbody></table></div>`;

        this.container.innerHTML = html;
        this.container.closest('.card').dataTable = this;
    }

    sort(columnIndex) {
        if (this.sortColumn === columnIndex) {
            this.sortAscending = !this.sortAscending;
        } else {
            this.sortColumn = columnIndex;
            this.sortAscending = true;
        }
        this.render();
    }

    filter(text) {
        this.filterText = text;
        this.render();
    }
}

// ファイルアップロード（ドラッグ&ドロップ対応）
function initFileUpload(dropZoneId, inputId, onUpload) {
    const dropZone = document.getElementById(dropZoneId);
    const input = document.getElementById(inputId);

    if (!dropZone || !input) return;

    dropZone.addEventListener('dragover', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--primary)';
        dropZone.style.background = 'rgba(102, 126, 234, 0.1)';
    });

    dropZone.addEventListener('dragleave', () => {
        dropZone.style.borderColor = 'var(--border)';
        dropZone.style.background = 'transparent';
    });

    dropZone.addEventListener('drop', (e) => {
        e.preventDefault();
        dropZone.style.borderColor = 'var(--border)';
        dropZone.style.background = 'transparent';

        const files = e.dataTransfer.files;
        if (files.length) {
            input.files = files;
            onUpload(files[0]);
        }
    });

    input.addEventListener('change', (e) => {
        if (e.target.files.length) {
            onUpload(e.target.files[0]);
        }
    });
}

// リアルタイム更新
class LiveUpdater {
    constructor(updateFn, interval = 10000) {
        this.updateFn = updateFn;
        this.interval = interval;
        this.timer = null;
    }

    start() {
        this.updateFn();
        this.timer = setInterval(() => this.updateFn(), this.interval);
    }

    stop() {
        if (this.timer) {
            clearInterval(this.timer);
            this.timer = null;
        }
    }
}

// ページ離脱時にタイマー停止
window.addEventListener('beforeunload', () => {
    if (window.liveUpdater) {
        window.liveUpdater.stop();
    }
});

// CSRF トークン取得
function getCookie(name) {
    let cookieValue = null;
    if (document.cookie && document.cookie !== '') {
        const cookies = document.cookie.split(';');
        for (let i = 0; i < cookies.length; i++) {
            const cookie = cookies[i].trim();
            if (cookie.substring(0, name.length + 1) === (name + '=')) {
                cookieValue = decodeURIComponent(cookie.substring(name.length + 1));
                break;
            }
        }
    }
    return cookieValue;
}

// API呼び出し（CSRFトークン付き）
async function apiCallWithCSRF(endpoint, options = {}) {
    const csrftoken = getCookie('csrftoken');
    const headers = {
        'Content-Type': 'application/json',
        'X-CSRFToken': csrftoken,
        ...options.headers
    };

    try {
        const res = await fetch(`${ChemML.API_URL}${endpoint}`, {
            ...options,
            headers
        });

        if (!res.ok) {
            const error = await res.json();
            throw new Error(error.detail || `HTTP ${res.status}`);
        }

        return await res.json();
    } catch (e) {
        showToast(e.message, 'error');
        throw e;
    }
}

// エクスポート
window.ChemML = ChemML;
window.ModalDialog = ModalDialog;
window.DataTable = DataTable;
window.initFileUpload = initFileUpload;
window.LiveUpdater = LiveUpdater;
window.apiCallWithCSRF = apiCallWithCSRF;
