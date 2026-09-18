/**
 * Predictive Fleet Intelligence — High-Performance Industrial Canvas Chart Engine
 * Renders smooth risk curves, prominent shaded Available Lead Time runways,
 * and synchronized crosshairs across multivariate sensor traces.
 */

class IndustrialChartEngine {
  constructor() {
    this.devicePixelRatio = window.devicePixelRatio || 1;
    this.riskData = null;
    this.sensorData = null;
    this.hoverCycle = null;
    this.listeners = [];
    this.zoomWindow = null; // null or { start: num, end: num }
  }

  // Helper to get theme colors dynamically
  getThemeColors() {
    const isDark = document.documentElement.getAttribute('data-theme') !== 'light';
    return {
      isDark,
      bg: isDark ? '#101F33' : '#EEF3F8',
      textPrimary: isDark ? '#F8FAFC' : '#122033',
      textSecondary: isDark ? '#94A3B8' : '#64748B',
      grid: isDark ? 'rgba(255, 255, 255, 0.07)' : 'rgba(0, 0, 0, 0.08)',
      riskLine: isDark ? '#00E5FF' : '#0284C7',
      baselineLine: isDark ? 'rgba(148, 163, 184, 0.5)' : 'rgba(100, 116, 139, 0.5)',
      thresholdLine: isDark ? '#94A3B8' : '#64748B',
      alertLine: '#F59E0B',
      failureLine: '#EF4444',
      leadTimeFill: isDark ? 'rgba(245, 158, 11, 0.22)' : 'rgba(245, 158, 11, 0.20)',
      leadTimeBorder: isDark ? 'rgba(245, 158, 11, 0.85)' : 'rgba(217, 119, 6, 0.85)',
      leadTimeBadgeBg: isDark ? 'rgba(11, 23, 40, 0.92)' : 'rgba(255, 255, 255, 0.94)',
      leadTimeText: isDark ? '#FBBF24' : '#B45309',
      crosshair: isDark ? '#00E5FF' : '#0284C7',
      sensorGreen: isDark ? '#10B981' : '#059669',
    };
  }

  setupCanvas(canvas) {
    const rect = canvas.getBoundingClientRect();
    canvas.width = rect.width * this.devicePixelRatio;
    canvas.height = rect.height * this.devicePixelRatio;
    const ctx = canvas.getContext('2d');
    ctx.scale(this.devicePixelRatio, this.devicePixelRatio);
    return { ctx, width: rect.width, height: rect.height };
  }

  // ----------------------------------------------------------------------------
  // MAIN RISK CHART
  // ----------------------------------------------------------------------------
  renderRiskChart(canvasId, timelineData) {
    this.riskData = timelineData;
    const canvas = document.getElementById(canvasId);
    if (!canvas || !timelineData || !timelineData.series || timelineData.series.length === 0) return;

    const { ctx, width, height } = this.setupCanvas(canvas);
    const colors = this.getThemeColors();

    const padding = { top: 35, right: 40, bottom: 45, left: 55 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const series = timelineData.series;
    const minCycle = this.zoomWindow ? this.zoomWindow.start : series[0].cycle;
    const maxCycle = this.zoomWindow ? this.zoomWindow.end : series[series.length - 1].cycle;

    const getX = (cycle) => padding.left + ((cycle - minCycle) / (maxCycle - minCycle)) * chartW;
    const getY = (val) => padding.top + (1.0 - val) * chartH;

    // Clear
    ctx.clearRect(0, 0, width, height);

    // 1. Grid Lines & Axis Ticks
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1;
    ctx.font = '10px JetBrains Mono, monospace';
    ctx.fillStyle = colors.textSecondary;
    ctx.textAlign = 'right';

    // Y Axis (0.0, 0.25, 0.50, 0.75, 1.0)
    const yTicks = [0.0, 0.25, 0.5, 0.75, 1.0];
    yTicks.forEach(tick => {
      const y = getY(tick);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      ctx.fillText(tick.toFixed(2), padding.left - 8, y + 4);
    });

    // Y Axis Title
    ctx.save();
    ctx.translate(16, height / 2);
    ctx.rotate(-Math.PI / 2);
    ctx.textAlign = 'center';
    ctx.font = '11px Inter, sans-serif';
    ctx.fillStyle = colors.textSecondary;
    ctx.fillText('Failure Risk [0.0 → 1.0]', 0, 0);
    ctx.restore();

    // X Axis Ticks (Cycles)
    ctx.textAlign = 'center';
    const cycleStep = Math.max(10, Math.round((maxCycle - minCycle) / 8 / 10) * 10);
    for (let c = Math.ceil(minCycle / cycleStep) * cycleStep; c <= maxCycle; c += cycleStep) {
      const x = getX(c);
      ctx.beginPath();
      ctx.moveTo(x, padding.top);
      ctx.lineTo(x, height - padding.bottom);
      ctx.stroke();
      ctx.fillText(`C-${c}`, x, height - padding.bottom + 18);
    }

    // X Axis Title
    ctx.font = '11px Inter, sans-serif';
    ctx.fillText('Operating Cycles (Time Progression)', padding.left + chartW / 2, height - 10);

    // 2. CRITICAL FEATURE: SHADED AVAILABLE LEAD TIME REGION
    const alertCycle = timelineData.first_alert_cycle;
    const failureCycle = timelineData.actual_failure_cycle;

    if (alertCycle && failureCycle && alertCycle < failureCycle) {
      const xAlert = getX(alertCycle);
      const xFailure = getX(failureCycle);

      // Shaded Region
      ctx.fillStyle = colors.leadTimeFill;
      ctx.fillRect(xAlert, padding.top, xFailure - xAlert, chartH);

      // Subtle hatch pattern in lead time region
      ctx.strokeStyle = 'rgba(245, 158, 11, 0.12)';
      ctx.lineWidth = 1;
      const step = 12;
      for (let x = xAlert; x <= xFailure + chartH; x += step) {
        ctx.beginPath();
        ctx.moveTo(Math.max(xAlert, x - chartH), padding.top + Math.min(chartH, chartH - (x - chartH - xAlert)));
        ctx.lineTo(Math.min(xFailure, x), padding.top);
        ctx.stroke();
      }

      // Vertical Alert Line (Amber)
      ctx.strokeStyle = colors.alertLine;
      ctx.lineWidth = 2.5;
      ctx.setLineDash([6, 4]);
      ctx.beginPath();
      ctx.moveTo(xAlert, padding.top);
      ctx.lineTo(xAlert, height - padding.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      // Vertical Failure Line (Red)
      ctx.strokeStyle = colors.failureLine;
      ctx.lineWidth = 2.5;
      ctx.beginPath();
      ctx.moveTo(xFailure, padding.top);
      ctx.lineTo(xFailure, height - padding.bottom);
      ctx.stroke();

      // Top Pins for Alert & Failure
      this.drawPin(ctx, xAlert, padding.top - 8, 'ALERT', colors.alertLine, '#FFFFFF');
      this.drawPin(ctx, xFailure, padding.top - 8, 'FAILURE', colors.failureLine, '#FFFFFF');

      // VISIBLE IN-REGION CALLOUT: "AVAILABLE LEAD TIME"
      const midX = (xAlert + xFailure) / 2;
      const badgeY = padding.top + chartH * 0.42;

      // Draw Badge Container
      const badgeW = 200;
      const badgeH = 58;
      ctx.fillStyle = colors.leadTimeBadgeBg;
      ctx.strokeStyle = colors.leadTimeBorder;
      ctx.lineWidth = 1.5;
      this.roundRect(ctx, midX - badgeW / 2, badgeY - badgeH / 2, badgeW, badgeH, 6, true, true);

      // Badge Text
      ctx.textAlign = 'center';
      ctx.fillStyle = colors.leadTimeText;
      ctx.font = '800 11px Inter, sans-serif';
      ctx.fillText('AVAILABLE LEAD TIME', midX, badgeY - 12);

      ctx.fillStyle = colors.textPrimary;
      ctx.font = '700 16px JetBrains Mono, monospace';
      ctx.fillText(`${timelineData.lead_time_cycles} cycles (${timelineData.lead_time_hours_str})`, midX, badgeY + 8);

      ctx.fillStyle = colors.textSecondary;
      ctx.font = '500 10px Inter, sans-serif';
      ctx.fillText('Intervention Runway to Service', midX, badgeY + 22);
    }

    // 3. Horizontal Operational Risk Threshold (0.50)
    const thresholdY = getY(timelineData.threshold || 0.50);
    ctx.strokeStyle = colors.thresholdLine;
    ctx.lineWidth = 1.8;
    ctx.setLineDash([5, 5]);
    ctx.beginPath();
    ctx.moveTo(padding.left, thresholdY);
    ctx.lineTo(width - padding.right, thresholdY);
    ctx.stroke();
    ctx.setLineDash([]);

    // Threshold label
    ctx.fillStyle = colors.textSecondary;
    ctx.font = '600 10px JetBrains Mono, monospace';
    ctx.textAlign = 'left';
    ctx.fillText('Risk Threshold (τ = 0.50)', padding.left + 8, thresholdY - 6);

    // 4. Baseline Risk Trajectory (Subtle benchmark line)
    ctx.strokeStyle = colors.baselineLine;
    ctx.lineWidth = 1.5;
    ctx.setLineDash([3, 3]);
    ctx.beginPath();
    let startedBase = false;
    series.forEach(pt => {
      if (pt.cycle >= minCycle && pt.cycle <= maxCycle) {
        const x = getX(pt.cycle);
        const y = getY(pt.risk_baseline);
        if (!startedBase) { ctx.moveTo(x, y); startedBase = true; }
        else { ctx.lineTo(x, y); }
      }
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // 5. LSTM Risk Trajectory Curve (Smooth Primary Curve)
    ctx.strokeStyle = colors.riskLine;
    ctx.lineWidth = 3.0;
    ctx.shadowColor = colors.riskLine;
    ctx.shadowBlur = colors.isDark ? 8 : 2;

    ctx.beginPath();
    let started = false;
    series.forEach(pt => {
      if (pt.cycle >= minCycle && pt.cycle <= maxCycle) {
        const x = getX(pt.cycle);
        const y = getY(pt.risk_lstm);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else { ctx.lineTo(x, y); }
      }
    });
    ctx.stroke();
    ctx.shadowBlur = 0; // reset shadow

    // 6. Draw Synchronized Crosshair if Hovered
    if (this.hoverCycle && this.hoverCycle >= minCycle && this.hoverCycle <= maxCycle) {
      const hX = getX(this.hoverCycle);
      ctx.strokeStyle = colors.crosshair;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 4]);
      ctx.beginPath();
      ctx.moveTo(hX, padding.top);
      ctx.lineTo(hX, height - padding.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      // Highlight active point on line
      const activePt = series.find(p => p.cycle === this.hoverCycle);
      if (activePt) {
        const ptY = getY(activePt.risk_lstm);
        ctx.fillStyle = colors.riskLine;
        ctx.beginPath();
        ctx.arc(hX, ptY, 5, 0, Math.PI * 2);
        ctx.fill();
        ctx.strokeStyle = '#FFFFFF';
        ctx.lineWidth = 2;
        ctx.stroke();
      }
    }
  }

  // ----------------------------------------------------------------------------
  // SYNCHRONIZED SENSOR TRACES
  // ----------------------------------------------------------------------------
  renderSensorChart(canvasId, sensorItem, alertCycle) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !sensorItem || !sensorItem.series || sensorItem.series.length === 0) return;

    const { ctx, width, height } = this.setupCanvas(canvas);
    const colors = this.getThemeColors();

    const padding = { top: 15, right: 25, bottom: 25, left: 45 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const series = sensorItem.series;
    const minCycle = this.zoomWindow ? this.zoomWindow.start : series[0].cycle;
    const maxCycle = this.zoomWindow ? this.zoomWindow.end : series[series.length - 1].cycle;

    const values = series.map(s => s.value);
    const minVal = Math.min(...values);
    const maxVal = Math.max(...values);
    const valRange = (maxVal - minVal) || 1;

    const getX = (cycle) => padding.left + ((cycle - minCycle) / (maxCycle - minCycle)) * chartW;
    const getY = (v) => padding.top + chartH - ((v - minVal) / valRange) * chartH;

    ctx.clearRect(0, 0, width, height);

    // Horizontal Grid
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1;
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillStyle = colors.textSecondary;
    ctx.textAlign = 'right';

    [minVal, (minVal + maxVal) / 2, maxVal].forEach(v => {
      const y = getY(v);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      ctx.fillText(v.toFixed(1), padding.left - 6, y + 3);
    });

    // Vertical Alert Reference
    if (alertCycle && alertCycle >= minCycle && alertCycle <= maxCycle) {
      const ax = getX(alertCycle);
      ctx.strokeStyle = colors.alertLine;
      ctx.lineWidth = 1.5;
      ctx.setLineDash([4, 3]);
      ctx.beginPath();
      ctx.moveTo(ax, padding.top);
      ctx.lineTo(ax, height - padding.bottom);
      ctx.stroke();
      ctx.setLineDash([]);
    }

    // Sensor Signal Line
    const lineColor = sensorItem.direction === 'up' ? '#F59E0B' : '#38BDF8';
    ctx.strokeStyle = lineColor;
    ctx.lineWidth = 1.8;
    ctx.beginPath();
    let started = false;
    series.forEach(pt => {
      if (pt.cycle >= minCycle && pt.cycle <= maxCycle) {
        const x = getX(pt.cycle);
        const y = getY(pt.value);
        if (!started) { ctx.moveTo(x, y); started = true; }
        else { ctx.lineTo(x, y); }
      }
    });
    ctx.stroke();

    // Synchronized Crosshair
    if (this.hoverCycle && this.hoverCycle >= minCycle && this.hoverCycle <= maxCycle) {
      const hx = getX(this.hoverCycle);
      ctx.strokeStyle = colors.crosshair;
      ctx.lineWidth = 1.2;
      ctx.setLineDash([3, 3]);
      ctx.beginPath();
      ctx.moveTo(hx, padding.top);
      ctx.lineTo(hx, height - padding.bottom);
      ctx.stroke();
      ctx.setLineDash([]);

      const activePt = series.find(p => p.cycle === this.hoverCycle);
      if (activePt) {
        const py = getY(activePt.value);
        ctx.fillStyle = lineColor;
        ctx.beginPath();
        ctx.arc(hx, py, 4, 0, Math.PI * 2);
        ctx.fill();
      }
    }
  }

  // ----------------------------------------------------------------------------
  // THRESHOLD SWEEP SENSITIVITY CHART (SCREEN 4)
  // ----------------------------------------------------------------------------
  renderSweepChart(canvasId, sweepData, selectedThreshold = 0.50) {
    const canvas = document.getElementById(canvasId);
    if (!canvas || !sweepData || sweepData.length === 0) return;

    const { ctx, width, height } = this.setupCanvas(canvas);
    const colors = this.getThemeColors();

    const padding = { top: 25, right: 35, bottom: 35, left: 45 };
    const chartW = width - padding.left - padding.right;
    const chartH = height - padding.top - padding.bottom;

    const getX = (t) => padding.left + ((t - 0.10) / (0.90 - 0.10)) * chartW;
    const getY = (v) => padding.top + (1.0 - v) * chartH;

    ctx.clearRect(0, 0, width, height);

    // Y Axis Grid
    ctx.strokeStyle = colors.grid;
    ctx.lineWidth = 1;
    ctx.font = '9px JetBrains Mono, monospace';
    ctx.fillStyle = colors.textSecondary;
    ctx.textAlign = 'right';

    [0.0, 0.25, 0.50, 0.75, 1.0].forEach(tick => {
      const y = getY(tick);
      ctx.beginPath();
      ctx.moveTo(padding.left, y);
      ctx.lineTo(width - padding.right, y);
      ctx.stroke();
      ctx.fillText(tick.toFixed(2), padding.left - 6, y + 3);
    });

    // X Axis Ticks
    ctx.textAlign = 'center';
    [0.1, 0.3, 0.5, 0.7, 0.9].forEach(t => {
      const x = getX(t);
      ctx.fillText(t.toFixed(2), x, height - 15);
    });

    // Plot Precision (Green)
    this.drawSeries(ctx, sweepData, d => getX(d.threshold), d => getY(d.precision), '#10B981', 2);
    // Plot Recall (Blue)
    this.drawSeries(ctx, sweepData, d => getX(d.threshold), d => getY(d.recall), '#38BDF8', 2);
    // Plot F1 (Purple)
    this.drawSeries(ctx, sweepData, d => getX(d.threshold), d => getY(d.f1), '#A855F7', 1.8, [4, 4]);

    // Operational Selection Marker (0.50)
    const selX = getX(selectedThreshold);
    ctx.strokeStyle = colors.alertLine;
    ctx.lineWidth = 2;
    ctx.setLineDash([5, 4]);
    ctx.beginPath();
    ctx.moveTo(selX, padding.top);
    ctx.lineTo(selX, height - padding.bottom);
    ctx.stroke();
    ctx.setLineDash([]);

    // Legend
    ctx.font = '10px Inter, sans-serif';
    ctx.fillStyle = '#10B981'; ctx.fillText('● Precision', padding.left + 40, padding.top - 8);
    ctx.fillStyle = '#38BDF8'; ctx.fillText('● Recall', padding.left + 120, padding.top - 8);
    ctx.fillStyle = '#A855F7'; ctx.fillText('▲ F1', padding.left + 190, padding.top - 8);
    ctx.fillStyle = colors.alertLine; ctx.fillText('● Operating (0.50)', padding.left + 280, padding.top - 8);
  }

  drawSeries(ctx, data, xFn, yFn, color, width, dash = []) {
    ctx.strokeStyle = color;
    ctx.lineWidth = width;
    ctx.setLineDash(dash);
    ctx.beginPath();
    data.forEach((d, idx) => {
      const x = xFn(d);
      const y = yFn(d);
      if (idx === 0) ctx.moveTo(x, y);
      else ctx.lineTo(x, y);
    });
    ctx.stroke();
    ctx.setLineDash([]);

    // Dots
    ctx.fillStyle = color;
    data.forEach(d => {
      ctx.beginPath();
      ctx.arc(xFn(d), yFn(d), 3, 0, Math.PI * 2);
      ctx.fill();
    });
  }

  drawPin(ctx, x, y, text, bgColor, textColor) {
    ctx.font = '800 9px JetBrains Mono, monospace';
    const textW = ctx.measureText(text).width;
    const pad = 6;
    const w = textW + pad * 2;
    const h = 16;

    ctx.fillStyle = bgColor;
    this.roundRect(ctx, x - w / 2, y - h, w, h, 3, true, false);

    ctx.fillStyle = textColor;
    ctx.textAlign = 'center';
    ctx.fillText(text, x, y - 4);
  }

  roundRect(ctx, x, y, width, height, radius, fill, stroke) {
    ctx.beginPath();
    ctx.moveTo(x + radius, y);
    ctx.lineTo(x + width - radius, y);
    ctx.quadraticCurveTo(x + width, y, x + width, y + radius);
    ctx.lineTo(x + width, y + height - radius);
    ctx.quadraticCurveTo(x + width, y + height, x + width - radius, y + height);
    ctx.lineTo(x + radius, y + height);
    ctx.quadraticCurveTo(x, y + height, x, y + height - radius);
    ctx.lineTo(x, y + radius);
    ctx.quadraticCurveTo(x, y, x + radius, y);
    ctx.closePath();
    if (fill) ctx.fill();
    if (stroke) ctx.stroke();
  }

  setHoverCycle(cycle) {
    this.hoverCycle = cycle;
  }
}

window.industrialChartEngine = new IndustrialChartEngine();
