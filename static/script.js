// ==================== Global State ====================
let currentMode = null;
let selectedFile = null;
let isProcessing = false;
let eventSource = null;
let alertsEnabled = true;

// Web Audio API context for generating beeps directly in the browser
let audioCtx = null;

function initAudio() {
    if (!audioCtx) {
        audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    }
    if (audioCtx.state === 'suspended') {
        audioCtx.resume();
    }
}

function playBeep(frequency, durationMs) {
    if (!alertsEnabled) return;
    initAudio();
    if (!audioCtx) return;

    const oscillator = audioCtx.createOscillator();
    const gainNode = audioCtx.createGain();

    oscillator.type = 'sine';
    oscillator.frequency.value = frequency;

    // Fade out to avoid clicks
    gainNode.gain.setValueAtTime(1, audioCtx.currentTime);
    gainNode.gain.exponentialRampToValueAtTime(0.001, audioCtx.currentTime + durationMs / 1000);

    oscillator.connect(gainNode);
    gainNode.connect(audioCtx.destination);

    oscillator.start();
    oscillator.stop(audioCtx.currentTime + durationMs / 1000);
}

// ==================== Mode Selection ====================
function selectMode(mode) {
    currentMode = mode;

    document.getElementById('modeSelection').classList.add('hidden');

    if (mode === 'upload') {
        document.getElementById('uploadPanel').classList.remove('hidden');
    } else if (mode === 'mobile') {
        document.getElementById('mobilePanel').classList.remove('hidden');
    }
}

function goBack() {
    // Clean up any active stream without triggering stop loop
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    isProcessing = false;

    document.getElementById('uploadPanel').classList.add('hidden');
    document.getElementById('mobilePanel').classList.add('hidden');
    document.getElementById('processingPanel').classList.add('hidden');
    document.getElementById('modeSelection').classList.remove('hidden');

    // Reset state
    removeFile();
}

// ==================== File Handling ====================
function handleFileSelect(event) {
    const file = event.target.files[0];
    if (file) {
        selectedFile = file;
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('selectedFile').classList.remove('hidden');
        document.getElementById('uploadArea').classList.add('hidden');
        document.getElementById('startUploadBtn').disabled = false;
    }
}

function removeFile() {
    selectedFile = null;
    document.getElementById('videoInput').value = '';
    document.getElementById('selectedFile').classList.add('hidden');
    document.getElementById('uploadArea').classList.remove('hidden');
    document.getElementById('startUploadBtn').disabled = true;
}

// ==================== Drag and Drop ====================
document.addEventListener('DOMContentLoaded', function () {
    const uploadArea = document.getElementById('uploadArea');

    if (uploadArea) {
        ['dragenter', 'dragover', 'dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, preventDefaults, false);
        });

        ['dragenter', 'dragover'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.add('dragover');
            });
        });

        ['dragleave', 'drop'].forEach(eventName => {
            uploadArea.addEventListener(eventName, () => {
                uploadArea.classList.remove('dragover');
            });
        });

        uploadArea.addEventListener('drop', handleDrop);
    }
});

function preventDefaults(e) {
    e.preventDefault();
    e.stopPropagation();
}

function handleDrop(e) {
    const dt = e.dataTransfer;
    const file = dt.files[0];

    if (file && file.type.startsWith('video/')) {
        selectedFile = file;
        document.getElementById('fileName').textContent = file.name;
        document.getElementById('selectedFile').classList.remove('hidden');
        document.getElementById('uploadArea').classList.add('hidden');
        document.getElementById('startUploadBtn').disabled = false;
    }
}

// ==================== Processing ====================
async function startProcessing(mode) {
    // Ensure clean state - stop any existing processing first
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }
    await fetch('/api/stop', { method: 'POST' }).catch(() => { });

    isProcessing = true;
    
    // Initialize audio context on first user interaction (browser policy requires this)
    initAudio();

    // Hide current panel, show processing panel
    document.getElementById('uploadPanel').classList.add('hidden');
    document.getElementById('mobilePanel').classList.add('hidden');
    document.getElementById('processingPanel').classList.remove('hidden');

    // Show loading overlay
    document.getElementById('videoOverlay').classList.remove('hidden');

    try {
        let response;

        if (mode === 'upload' && selectedFile) {
            // Upload the video file
            const formData = new FormData();
            formData.append('video', selectedFile);

            response = await fetch('/api/start', {
                method: 'POST',
                body: formData
            });
        } else if (mode === 'mobile') {
            // Start with mobile camera
            const cameraIndex = document.getElementById('cameraIndex').value;

            response = await fetch('/api/start', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({
                    mode: 'camera',
                    camera_index: parseInt(cameraIndex)
                })
            });
        }

        if (response && response.ok) {
            // Keep overlay visible until first frame actually arrives
            // document.getElementById('videoOverlay').classList.add('hidden'); // Removed here
            startVideoStream();
        } else {
            throw new Error('Failed to start processing');
        }
    } catch (error) {
        console.error('Error starting processing:', error);
        alert('Failed to start processing. Make sure the server is running.');
        goBack();
    }
}

function startVideoStream() {
    let frameReceived = false;
    let connectionTimeout = setTimeout(() => {
        if (!frameReceived && isProcessing) {
            console.error('Camera connection timeout');
            alert('Camera connection timed out. No frames received. Check iVCam connection.');
            stopProcessing();
        }
    }, 15000); // 15s timeout

    // Use Server-Sent Events for real-time updates
    eventSource = new EventSource('/api/stream');

    eventSource.onmessage = function (event) {
        const data = JSON.parse(event.data);

        // Update connection status text if provided
        if (data.statusMessage) {
            const statusEl = document.getElementById('connectionStatus');
            if (statusEl) statusEl.textContent = data.statusMessage;
        }

        // Handle error message from server
        if (data.errorMessage) {
            alert(data.errorMessage);
            stopProcessing();
            return;
        }

        // Hide overlay on first frame
        if (data.frame && !frameReceived) {
            frameReceived = true;
            clearTimeout(connectionTimeout);
            document.getElementById('videoOverlay').classList.add('hidden');
        }

        // Handle session complete - show summary modal
        if (data.session_complete && data.summary) {
            showSummary(data.summary);
            return;
        }

        // Handle stream stopped
        if (data.status === 'stopped') {
            if (eventSource) {
                eventSource.close();
                eventSource = null;
            }
            return;
        }

        // Update video feed
        if (data.frame) {
            document.getElementById('videoFeed').src = 'data:image/jpeg;base64,' + data.frame;
        }

        // Update pothole status
        if (data.pothole_detected !== undefined) {
            const potholeValue = document.getElementById('potholeStatus');
            const potholeCard = potholeValue.closest('.stat-card');
            if (data.pothole_detected) {
                potholeValue.textContent = 'POTHOLE!';
                potholeValue.classList.add('warning');
                potholeValue.classList.remove('safe');
                potholeCard.classList.add('alert');
                
                // Play high-pitched beep for longer duration
                if (!window.lastPotholeBeep || (Date.now() - window.lastPotholeBeep > 800)) {
                    playBeep(1800, 450);  // Increased from 150ms to 450ms
                    window.lastPotholeBeep = Date.now();
                }
            } else {
                potholeValue.textContent = 'Safe';
                potholeValue.classList.remove('warning');
                potholeValue.classList.add('safe');
                potholeCard.classList.remove('alert');
            }
        }

        if (data.lane_status) {
            const laneValue = document.getElementById('laneStatus');
            const laneCard = laneValue.closest('.stat-card');
            laneValue.textContent = data.lane_status;
            if (data.lane_status === 'DEPARTURE') {
                laneValue.classList.add('warning');
                laneCard.classList.add('alert');
                
                // Play lower-pitched beep smoothly for longer duration
                if (!window.lastLaneBeep || (Date.now() - window.lastLaneBeep > 1000)) {
                    playBeep(1200, 600);  // Increased from 200ms to 600ms
                    window.lastLaneBeep = Date.now();
                }
            } else {
                laneValue.classList.remove('warning');
                laneCard.classList.remove('alert');
            }
        }

        if (data.offset !== undefined) {
            document.getElementById('offsetValue').textContent = data.offset.toFixed(2) + 'm';
        }

        // Update live pothole count
        if (data.pothole_count !== undefined) {
            document.getElementById('potholeCount').textContent = data.pothole_count;
        }
    };

    eventSource.onerror = function () {
        console.log('Stream ended or error');
        if (eventSource) {
            eventSource.close();
            eventSource = null;
        }
        if (isProcessing) {
            isProcessing = false;
            // Send a single stop request to clean up server-side
            fetch('/api/stop', { method: 'POST' }).catch(() => { });
            // Don't go back if summary modal is showing
            if (document.getElementById('summaryOverlay').classList.contains('hidden')) {
                document.getElementById('processingPanel').classList.add('hidden');
                document.getElementById('modeSelection').classList.remove('hidden');
            }
        }
    };
}

// ==================== Session Summary ====================
function showSummary(summary) {
    document.getElementById('summaryTotalPotholes').textContent = summary.total_potholes || 0;

    if (summary.total_potholes > 0) {
        document.getElementById('summaryAvgConf').textContent = (summary.avg_confidence * 100).toFixed(1) + '%';
        document.getElementById('summaryMinConf').textContent = (summary.min_confidence * 100).toFixed(1) + '%';
        document.getElementById('summaryMaxConf').textContent = (summary.max_confidence * 100).toFixed(1) + '%';
    } else {
        document.getElementById('summaryAvgConf').textContent = '—';
        document.getElementById('summaryMinConf').textContent = '—';
        document.getElementById('summaryMaxConf').textContent = '—';
    }

    document.getElementById('summaryOverlay').classList.remove('hidden');
}

function closeSummary() {
    document.getElementById('summaryOverlay').classList.add('hidden');
    isProcessing = false;

    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    goBack();
}

async function stopProcessing() {
    // Guard against re-entry
    if (!isProcessing && !eventSource) return;

    isProcessing = false;

    // Close event source
    if (eventSource) {
        eventSource.close();
        eventSource = null;
    }

    // Tell server to stop
    try {
        await fetch('/api/stop', { method: 'POST' });
    } catch (error) {
        console.error('Error stopping:', error);
    }

    // Reset stats
    const potholeStatus = document.getElementById('potholeStatus');
    if (potholeStatus) {
        potholeStatus.textContent = 'Safe';
        potholeStatus.classList.remove('warning');
        potholeStatus.classList.add('safe');
        potholeStatus.closest('.stat-card').classList.remove('alert');
    }
    document.getElementById('laneStatus').textContent = 'Normal';
    document.getElementById('laneStatus').classList.remove('warning');
    document.getElementById('offsetValue').textContent = '0.00m';
    document.getElementById('potholeCount').textContent = '0';

    // Go back to main screen
    document.getElementById('processingPanel').classList.add('hidden');
    document.getElementById('modeSelection').classList.remove('hidden');
}

// ==================== Alert Toggles ====================
async function toggleAlerts() {
    alertsEnabled = !alertsEnabled;
    const btn = document.getElementById('muteBtn');
    const icon = document.getElementById('muteIcon');
    const text = document.getElementById('muteText');

    if (alertsEnabled) {
        btn.classList.remove('muted');
        text.textContent = 'Alerts On';
        icon.innerHTML = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" /><path d="M19.07 4.93a10 10 0 0 1 0 14.14M15.54 8.46a5 5 0 0 1 0 7.07" />';
    } else {
        btn.classList.add('muted');
        text.textContent = 'Alerts Off';
        icon.innerHTML = '<polygon points="11 5 6 9 2 9 2 15 6 15 11 19 11 5" /><line x1="23" y1="9" x2="17" y2="15" /><line x1="17" y1="9" x2="23" y2="15" />';
    }

    try {
        await fetch('/api/toggle_alerts', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify({ enabled: alertsEnabled })
        });
    } catch (e) {
        console.error('Error toggling alerts state on server:', e);
    }
}

// ==================== Keyboard Shortcuts ====================
document.addEventListener('keydown', function (e) {
    // ESC to go back or stop
    if (e.key === 'Escape') {
        if (isProcessing) {
            stopProcessing();
        } else {
            goBack();
        }
    }
});
