/**
 * Predictive Fleet Intelligence — Industrial Application Controller
 * Handles live telemetry streams, API integration, synchronized crosshairs,
 * fleet table filtering, and dual-mode theme management.
 */

(function () {
  'use strict';

  const AppState = {
    selectedMachineId: 48, // Default star machine for demo
    activeTab: 'screen-risk',
    theme: localStorage.getItem('fleet_theme') || 'dark',
    liveTelemetry: true,
    fleet: null,
    timeline: null,
    sensors: null,
    metrics: null,
    alerts: null,
    activeFilter: 'all',
    searchQuery: '',
  };

  const API = {
    async getHealth() {
      const res = await fetch('/api/health');
      return res.json();
    },
    async getFleet() {
      const res = await fetch('/api/fleet');
      return res.json();
    },
    async getTimeline(machineId) {
      const res = await fetch(`/api/machines/${machineId}/timeline`);
      return res.json();
    },
    async getSensors(machineId) {
      const res = await fetch(`/api/machines/${machineId}/sensors`);
      return res.json();
    },
    async getModelComparison() {
      const res = await fetch('/api/model-comparison');
      return res.json();
    },
    async getThresholdSweep() {
      const res = await fetch('/api/threshold-sweep');
      return res.json();
    },
    async getAlerts() {
      const res = await fetch('/api/alerts');
      return res.json();
    },
  };

  // ----------------------------------------------------------------------------
  // INITIALIZATION
  // ----------------------------------------------------------------------------
  async function init() {
    setupTheme();
    setupNavigation();
    setupControls();
    setupCrosshairInteraction();
    startLiveClock();

    try {
      await loadFleetData();
      await loadMachineTelemetry(AppState.selectedMachineId);
      await loadModelComparisonData();
      await loadAlertsData();
    } catch (err) {
      console.error('Initial telemetry fetch error:', err);
    }

    window.addEventListener('resize', debounce(() => {
      renderAllCharts();
    }, 150));
  }

  // ----------------------------------------------------------------------------
  // THEME MANAGEMENT (60% Dark / 40% Light Secondary Enterprise Mode)
  // ----------------------------------------------------------------------------
  function setupTheme() {
    document.documentElement.setAttribute('data-theme', AppState.theme);
    updateThemeLabel();

    const toggleSidebar = document.getElementById('theme-toggle-sidebar');
    const toggleHeader = document.getElementById('btn-theme-header');

    const toggleFn = () => {
      AppState.theme = AppState.theme === 'dark' ? 'light' : 'dark';
      document.documentElement.setAttribute('data-theme', AppState.theme);
      localStorage.setItem('fleet_theme', AppState.theme);
      updateThemeLabel();
      renderAllCharts();
    };

    if (toggleSidebar) toggleSidebar.addEventListener('click', toggleFn);
    if (toggleHeader) toggleHeader.addEventListener('click', toggleFn);
  }

  function updateThemeLabel() {
    const label = document.getElementById('theme-label-sidebar');
    if (label) {
      label.textContent = AppState.theme === 'dark' ? 'Light Mode' : 'Dark Mode';
    }
  }

  // ----------------------------------------------------------------------------
  // NAVIGATION & TAB SWITCHING
  // ----------------------------------------------------------------------------
  function setupNavigation() {
    const navItems = document.querySelectorAll('.nav-item[data-tab]');
    navItems.forEach(item => {
      item.addEventListener('click', () => {
        const targetTab = item.getAttribute('data-tab');
        switchTab(targetTab);
      });
    });

    // Showcase button for Machine 48
    const showcaseBtn = document.getElementById('btn-showcase-m48');
    if (showcaseBtn) {
      showcaseBtn.addEventListener('click', () => {
        selectMachine(48);
        switchTab('screen-risk');
      });
    }

    // Alerts Drawer toggles
    const openAlertsBtn = document.getElementById('btn-open-alerts');
    const headerAlertsBtn = document.getElementById('btn-header-alerts');
    const closeAlertsBtn = document.getElementById('btn-close-alerts');
    const alertsOverlay = document.getElementById('alerts-overlay');

    const openDrawer = () => {
      document.getElementById('alerts-drawer').classList.add('open');
      document.getElementById('alerts-overlay').classList.add('open');
    };
    const closeDrawer = () => {
      document.getElementById('alerts-drawer').classList.remove('open');
      document.getElementById('alerts-overlay').classList.remove('open');
    };

    if (openAlertsBtn) openAlertsBtn.addEventListener('click', openDrawer);
    if (headerAlertsBtn) headerAlertsBtn.addEventListener('click', openDrawer);
    if (closeAlertsBtn) closeAlertsBtn.addEventListener('click', closeDrawer);
    if (alertsOverlay) alertsOverlay.addEventListener('click', closeDrawer);
  }

  function switchTab(tabId) {
    AppState.activeTab = tabId;

    // Update nav active states
    document.querySelectorAll('.nav-item[data-tab]').forEach(btn => {
      btn.classList.toggle('active', btn.getAttribute('data-tab') === tabId);
    });

    // Update screen views
    document.querySelectorAll('.screen-view').forEach(screen => {
      screen.classList.toggle('active', screen.id === tabId);
    });

    // Update breadcrumb & title
    const screenNames = {
      'screen-risk': { title: 'Risk Monitor', sub: 'Machine failure early-warning intelligence and predictive runway analysis' },
      'screen-sensors': { title: 'Sensor Analytics', sub: 'Multivariate sensor signatures and cross-channel degradation drift traces' },
      'screen-fleet': { title: 'Fleet Overview', sub: 'Asset health status, priority rankings, and fleet-wide runway distribution' },
      'screen-models': { title: 'Model Performance', sub: 'Rigorous baseline vs. sequence model evaluation on held-out test engines' },
    };

    const info = screenNames[tabId] || screenNames['screen-risk'];
    document.getElementById('page-title').textContent = info.title;
    document.getElementById('page-subtitle').textContent = info.sub;
    document.getElementById('breadcrumb-screen').textContent = info.title;

    // Render charts on active screen
    setTimeout(() => {
      renderAllCharts();
    }, 50);
  }

  // ----------------------------------------------------------------------------
  // CONTROLS & MACHINE SELECTOR
  // ----------------------------------------------------------------------------
  function setupControls() {
    const machineSelect = document.getElementById('machine-select');
    if (machineSelect) {
      machineSelect.addEventListener('change', (e) => {
        selectMachine(parseInt(e.target.value, 10));
      });
    }

    const refreshBtn = document.getElementById('btn-manual-refresh');
    if (refreshBtn) {
      refreshBtn.addEventListener('click', async () => {
        refreshBtn.style.transform = 'rotate(180deg)';
        await loadMachineTelemetry(AppState.selectedMachineId);
        await loadFleetData();
        setTimeout(() => { refreshBtn.style.transform = 'none'; }, 300);
      });
    }

    const zoomLeadtimeBtn = document.getElementById('btn-zoom-leadtime');
    if (zoomLeadtimeBtn) {
      zoomLeadtimeBtn.addEventListener('click', () => {
        if (AppState.timeline && AppState.timeline.first_alert_cycle) {
          const alertC = AppState.timeline.first_alert_cycle;
          const failC = AppState.timeline.actual_failure_cycle || (alertC + 45);
          window.industrialChartEngine.zoomWindow = {
            start: Math.max(0, alertC - 15),
            end: failC + 5,
          };
          renderAllCharts();
        }
      });
    }

    const resetZoomBtn = document.getElementById('btn-reset-zoom');
    if (resetZoomBtn) {
      resetZoomBtn.addEventListener('click', () => {
        window.industrialChartEngine.zoomWindow = null;
        renderAllCharts();
      });
    }

    // Fleet search and filter buttons
    const searchInput = document.getElementById('fleet-search-input');
    if (searchInput) {
      searchInput.addEventListener('input', (e) => {
        AppState.searchQuery = e.target.value.toLowerCase().trim();
        renderFleetTable();
      });
    }

    const filterBtns = document.querySelectorAll('.filter-btn');
    filterBtns.forEach(btn => {
      btn.addEventListener('click', () => {
        filterBtns.forEach(b => b.classList.remove('active'));
        btn.classList.add('active');
        AppState.activeFilter = btn.getAttribute('data-filter');
        renderFleetTable();
      });
    });
  }

  // ----------------------------------------------------------------------------
  // DATA LOADING
  // ----------------------------------------------------------------------------
  async function loadFleetData() {
    try {
      const data = await API.getFleet();
      AppState.fleet = data;

      // Update KPI strip
      document.getElementById('kpi-fleet-total').textContent = data.kpis.total_machines;
      document.getElementById('kpi-fleet-healthy').textContent = data.kpis.healthy;
      document.getElementById('kpi-fleet-alert').textContent = data.kpis.at_risk;
      document.getElementById('kpi-fleet-critical').textContent = data.kpis.critical;
      document.getElementById('kpi-fleet-leadtime').textContent = `${data.kpis.mean_fleet_lead_time.toFixed(1)}c`;
      document.getElementById('fleet-count-badge').textContent = data.kpis.total_machines;

      // Populate machine selector
      const select = document.getElementById('machine-select');
      if (select) {
        select.innerHTML = '';
        data.machines.forEach(m => {
          const opt = document.createElement('option');
          opt.value = m.machine_id;
          opt.textContent = `${m.code} — ${m.name.split('(')[0].trim()} (${m.status})`;
          if (m.machine_id === AppState.selectedMachineId) opt.selected = true;
          select.appendChild(opt);
        });
      }

      renderFleetTable();
    } catch (err) {
      console.error('Failed to load fleet data:', err);
    }
  }

  async function loadMachineTelemetry(machineId) {
    try {
      const [timeline, sensors] = await Promise.all([
        API.getTimeline(machineId),
        API.getSensors(machineId),
      ]);

      AppState.timeline = timeline;
      AppState.sensors = sensors;

      // Update Breadcrumb & Top Bar
      document.getElementById('breadcrumb-machine').textContent = timeline.machine_code;
      const select = document.getElementById('machine-select');
      if (select) select.value = machineId;

      // Update Risk Monitor KPI Strip
      document.getElementById('val-current-risk').textContent = timeline.current_risk.toFixed(3);
      document.getElementById('val-failure-prob').textContent = `${timeline.current_failure_prob_pct}%`;
      document.getElementById('bar-risk').style.width = `${timeline.current_failure_prob_pct}%`;

      const statusEl = document.getElementById('val-status');
      statusEl.textContent = timeline.current_status === 'ALERT' ? '▲ ALERT' : (timeline.current_status === 'CRITICAL' ? '● CRITICAL' : '● HEALTHY');
      statusEl.className = `status-indicator-badge badge-${timeline.current_status.toLowerCase()}`;

      // Prominent Lead Time Values
      document.getElementById('val-lead-time').textContent = `${timeline.lead_time_cycles} cycles`;
      document.getElementById('val-lead-time-hours').textContent = timeline.lead_time_hours_str;

      // Update Sensor Readings in 2x2 Strip
      if (sensors && sensors.sensors) {
        sensors.sensors.forEach(s => {
          if (s.sensor_id === 'sensor_11') {
            document.getElementById('s11-val').textContent = s.current_value.toFixed(2);
            document.getElementById('s11-status').textContent = `${s.delta_pct >= 0 ? '▲' : '▼'} ${s.delta_pct}% ${s.status}`;
          } else if (s.sensor_id === 'sensor_9') {
            document.getElementById('s9-val').textContent = s.current_value.toFixed(1);
            document.getElementById('s9-status').textContent = `${s.delta_pct >= 0 ? '▲' : '▼'} ${s.delta_pct}% ${s.status}`;
          } else if (s.sensor_id === 'sensor_12') {
            document.getElementById('s12-val').textContent = s.current_value.toFixed(2);
            document.getElementById('s12-status').textContent = `${s.delta_pct >= 0 ? '▲' : '▼'} ${s.delta_pct}% ${s.status}`;
          } else if (s.sensor_id === 'sensor_14') {
            document.getElementById('s14-val').textContent = s.current_value.toFixed(1);
            document.getElementById('s14-status').textContent = `${s.delta_pct >= 0 ? '▲' : '▼'} ${s.delta_pct}% ${s.status}`;
          }
        });
      }

      const activeBadge = document.getElementById('sensor-active-machine-badge');
      if (activeBadge) {
        activeBadge.textContent = `Inspecting: ${timeline.machine_code} (Cycle ${timeline.current_cycle})`;
      }

      renderAllCharts();
    } catch (err) {
      console.error('Failed to load machine telemetry:', err);
    }
  }

  async function loadModelComparisonData() {
    try {
      const [compData, sweepData] = await Promise.all([
        API.getModelComparison(),
        API.getThresholdSweep(),
      ]);

      AppState.modelComparison = compData;
      AppState.thresholdSweep = sweepData;

      renderModelComparisonTable(compData.comparison);
      window.industrialChartEngine.renderSweepChart('sweep-chart-canvas', sweepData.sweep, 0.50);
    } catch (err) {
      console.error('Failed to load model comparison data:', err);
    }
  }

  async function loadAlertsData() {
    try {
      const alerts = await API.getAlerts();
      AppState.alerts = alerts;

      const listContainer = document.getElementById('drawer-alerts-list');
      if (!listContainer) return;

      listContainer.innerHTML = '';
      alerts.alerts.forEach(alt => {
        const item = document.createElement('div');
        item.className = `alert-ticket ${alt.severity === 'CRITICAL' ? 'ticket-critical' : ''}`;
        item.innerHTML = `
          <div class="ticket-header">
            <span class="ticket-code">${alt.machine_code} — ${alt.name}</span>
            <span class="ticket-runway">Runway: ${alt.available_lead_time}</span>
          </div>
          <div class="ticket-meta text-secondary">
            Severity: <strong class="${alt.severity === 'CRITICAL' ? 'text-red' : 'text-amber'}">${alt.severity}</strong> • Risk: <strong>${alt.risk_score}</strong> (${alt.failure_probability_pct}%)
          </div>
          <div class="ticket-driver text-muted" style="font-size:11px; margin-top:2px;">
            Driver: ${alt.primary_driver}
          </div>
          <div class="ticket-action-box">
            <strong>Recommended Action:</strong> ${alt.recommended_action}
          </div>
        `;
        item.addEventListener('click', () => {
          selectMachine(alt.machine_id);
          switchTab('screen-risk');
          document.getElementById('alerts-drawer').classList.remove('open');
          document.getElementById('alerts-overlay').classList.remove('open');
        });
        listContainer.appendChild(item);
      });

      document.getElementById('header-alert-count').textContent = alerts.total_active;
      document.getElementById('sidebar-alert-pill').textContent = `${alerts.total_active} Active`;
    } catch (err) {
      console.error('Failed to load alerts:', err);
    }
  }

  // ----------------------------------------------------------------------------
  // MACHINE SELECTION
  // ----------------------------------------------------------------------------
  function selectMachine(machineId) {
    AppState.selectedMachineId = machineId;
    window.industrialChartEngine.zoomWindow = null; // reset zoom
    loadMachineTelemetry(machineId);

    // Highlight row in fleet table
    document.querySelectorAll('#fleet-table-body tr').forEach(tr => {
      tr.classList.toggle('row-selected', parseInt(tr.getAttribute('data-id'), 10) === machineId);
    });
  }

  // ----------------------------------------------------------------------------
  // CHART RENDERING DISPATCHER
  // ----------------------------------------------------------------------------
  function renderAllCharts() {
    const engine = window.industrialChartEngine;
    if (AppState.timeline) {
      engine.renderRiskChart('risk-chart-canvas', AppState.timeline);
    }

    if (AppState.sensors && AppState.sensors.sensors) {
      const alertC = AppState.timeline ? AppState.timeline.first_alert_cycle : null;
      AppState.sensors.sensors.forEach(s => {
        if (s.sensor_id === 'sensor_11') engine.renderSensorChart('sensor-canvas-11', s, alertC);
        else if (s.sensor_id === 'sensor_9') engine.renderSensorChart('sensor-canvas-9', s, alertC);
        else if (s.sensor_id === 'sensor_12') engine.renderSensorChart('sensor-canvas-12', s, alertC);
        else if (s.sensor_id === 'sensor_14') engine.renderSensorChart('sensor-canvas-14', s, alertC);
      });
    }

    if (AppState.thresholdSweep) {
      engine.renderSweepChart('sweep-chart-canvas', AppState.thresholdSweep.sweep, 0.50);
    }
  }

  // ----------------------------------------------------------------------------
  // SYNCHRONIZED MOUSE CROSSHAIR INTERACTION
  // ----------------------------------------------------------------------------
  function setupCrosshairInteraction() {
    const riskCanvas = document.getElementById('risk-chart-canvas');
    const tooltip = document.getElementById('risk-tooltip');
    if (!riskCanvas || !tooltip) return;

    riskCanvas.addEventListener('mousemove', (e) => {
      if (!AppState.timeline || !AppState.timeline.series) return;

      const rect = riskCanvas.getBoundingClientRect();
      const mouseX = e.clientX - rect.left;
      const mouseY = e.clientY - rect.top;

      const padding = { left: 55, right: 40 };
      const chartW = rect.width - padding.left - padding.right;

      if (mouseX < padding.left || mouseX > rect.width - padding.right) {
        tooltip.style.display = 'none';
        window.industrialChartEngine.setHoverCycle(null);
        renderAllCharts();
        return;
      }

      const series = AppState.timeline.series;
      const minCycle = window.industrialChartEngine.zoomWindow ? window.industrialChartEngine.zoomWindow.start : series[0].cycle;
      const maxCycle = window.industrialChartEngine.zoomWindow ? window.industrialChartEngine.zoomWindow.end : series[series.length - 1].cycle;

      const fraction = (mouseX - padding.left) / chartW;
      const targetCycle = Math.round(minCycle + fraction * (maxCycle - minCycle));

      // Find nearest point
      const nearestPt = series.reduce((prev, curr) =>
        Math.abs(curr.cycle - targetCycle) < Math.abs(prev.cycle - targetCycle) ? curr : prev
      );

      // Update engine hover state & trigger synchronized re-render
      window.industrialChartEngine.setHoverCycle(nearestPt.cycle);
      renderAllCharts();

      // Show Tooltip
      tooltip.style.display = 'block';
      tooltip.style.left = `${Math.min(rect.width - 180, Math.max(10, mouseX + 14))}px`;
      tooltip.style.top = `${Math.min(rect.height - 110, Math.max(10, mouseY - 40))}px`;

      const statusColor = nearestPt.risk_lstm >= 0.50 ? 'var(--color-alert)' : 'var(--color-healthy)';

      tooltip.innerHTML = `
        <div class="tooltip-title">Operating Cycle ${nearestPt.cycle}</div>
        <div class="tooltip-row"><span>Risk Score:</span> <span class="tooltip-val" style="color:${statusColor}">${nearestPt.risk_lstm.toFixed(4)}</span></div>
        <div class="tooltip-row"><span>Status:</span> <span class="tooltip-val" style="color:${statusColor}">${nearestPt.status}</span></div>
        <div class="tooltip-row"><span>Failure Prob:</span> <span class="tooltip-val">${nearestPt.failure_probability}%</span></div>
        <div class="tooltip-row"><span>Baseline Risk:</span> <span class="tooltip-val text-muted">${nearestPt.risk_baseline.toFixed(3)}</span></div>
      `;
    });

    riskCanvas.addEventListener('mouseleave', () => {
      tooltip.style.display = 'none';
      window.industrialChartEngine.setHoverCycle(null);
      renderAllCharts();
    });
  }

  // ----------------------------------------------------------------------------
  // FLEET TABLE RENDERING
  // ----------------------------------------------------------------------------
  function renderFleetTable() {
    const tbody = document.getElementById('fleet-table-body');
    if (!tbody || !AppState.fleet || !AppState.fleet.machines) return;

    let list = [...AppState.fleet.machines];

    // Filter by tab
    if (AppState.activeFilter === 'alert') list = list.filter(m => m.status === 'ALERT');
    else if (AppState.activeFilter === 'critical') list = list.filter(m => m.status === 'CRITICAL');
    else if (AppState.activeFilter === 'healthy') list = list.filter(m => m.status === 'HEALTHY');

    // Search query
    if (AppState.searchQuery) {
      list = list.filter(m =>
        m.code.toLowerCase().includes(AppState.searchQuery) ||
        m.name.toLowerCase().includes(AppState.searchQuery) ||
        m.location.toLowerCase().includes(AppState.searchQuery) ||
        m.status.toLowerCase().includes(AppState.searchQuery)
      );
    }

    document.getElementById('table-filtered-counter').textContent = `Showing ${list.length} of ${AppState.fleet.machines.length} machines`;

    tbody.innerHTML = '';
    list.forEach(m => {
      const tr = document.createElement('tr');
      tr.setAttribute('data-id', m.machine_id);
      if (m.machine_id === AppState.selectedMachineId) tr.classList.add('row-selected');
      if (m.status === 'CRITICAL') tr.classList.add('row-critical');

      const statusIcon = m.status === 'HEALTHY' ? '● HEALTHY' : (m.status === 'ALERT' ? '▲ ALERT' : '● CRITICAL');
      const badgeClass = m.status.toLowerCase();
      const riskColor = m.status === 'CRITICAL' ? 'var(--color-critical)' : (m.status === 'ALERT' ? 'var(--color-alert)' : 'var(--color-healthy)');

      tr.innerHTML = `
        <td class="font-mono"><strong>#${m.priority_rank}</strong></td>
        <td>
          <div class="machine-cell">
            <span class="machine-code-strong">${m.code}</span>
            <span class="machine-name-sub">${m.name}</span>
          </div>
        </td>
        <td class="text-secondary">${m.location}</td>
        <td>
          <div class="risk-cell">
            <div class="risk-bar-mini">
              <div class="risk-bar-mini-fill" style="width: ${m.failure_probability_pct}%; background-color: ${riskColor};"></div>
            </div>
            <span class="font-mono" style="color: ${riskColor}; font-weight: 700;">${m.current_risk.toFixed(3)}</span>
          </div>
        </td>
        <td class="font-mono">${m.failure_probability_pct}%</td>
        <td>
          <span class="status-indicator-badge badge-${badgeClass}">${statusIcon}</span>
        </td>
        <td>
          <span class="table-leadtime-badge">${m.lead_time_display}</span>
        </td>
        <td>
          <span class="${m.risk_trend === 'rising' ? 'text-amber' : 'text-green'} font-bold">
            ${m.risk_trend === 'rising' ? '▲ Rising' : '● Stable'}
          </span>
        </td>
        <td>
          <button class="btn-subtle btn-inspect-row" data-id="${m.machine_id}">Inspect Risk</button>
        </td>
      `;

      tr.addEventListener('click', (e) => {
        selectMachine(m.machine_id);
        switchTab('screen-risk');
      });

      tbody.appendChild(tr);
    });
  }

  // ----------------------------------------------------------------------------
  // MODEL COMPARISON TABLE
  // ----------------------------------------------------------------------------
  function renderModelComparisonTable(comparisonRows) {
    const tbody = document.getElementById('model-comparison-tbody');
    if (!tbody || !comparisonRows) return;

    tbody.innerHTML = '';
    comparisonRows.forEach(row => {
      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><strong>${row.metric}</strong></td>
        <td class="font-mono text-muted">${row.baseline}</td>
        <td class="font-mono font-bold text-cyan">${row.lstm}</td>
        <td><span class="advantage-badge">${row.advantage}</span></td>
        <td class="text-secondary" style="font-size:11px;">${row.description}</td>
      `;
      tbody.appendChild(tr);
    });
  }

  // ----------------------------------------------------------------------------
  // LIVE TELEMETRY CLOCK
  // ----------------------------------------------------------------------------
  function startLiveClock() {
    const clockEl = document.getElementById('telemetry-clock');
    const updateTime = () => {
      const now = new Date();
      if (clockEl) {
        clockEl.textContent = now.toLocaleTimeString('en-US', { hour12: true });
      }
    };
    updateTime();
    setInterval(updateTime, 1000);
  }

  // Utility: debounce
  function debounce(fn, wait) {
    let timer;
    return (...args) => {
      clearTimeout(timer);
      timer = setTimeout(() => fn(...args), wait);
    };
  }

  // Start application on DOM Ready
  if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
  } else {
    init();
  }

})();
