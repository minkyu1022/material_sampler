/* Interactive 2 — live 1D Bridge Matching Sampler.
   Faithful to src/bms: independent coupling + the target-score identity
   nelson = gamma_t*(prior_score(X0') + target_score(X1)) - (X0'-X1)/kappa1,
   constant-diffusion Brownian reference, Markovianization via Nadaraya-Watson
   conditional-expectation regression, damped fixed-point update (eta). */
(function () {
  const field = document.getElementById('fieldCanvas');
  const hist = document.getElementById('histCanvas');
  if (!field || !hist) return;
  const fx = field.getContext('2d'), hx = hist.getContext('2d');

  // ---- constants ----
  const sigma_p = 1.0, g = 1.3, kappa1 = g * g;
  const L = 6.0, Nx = 101, Nt = 25, B = 2200;
  const dx = 2 * L / (Nx - 1), dt = 1 / (Nt - 1);
  const h = 1.8 * dx, halfW = Math.ceil(4 * h / dx);
  const xgrid = new Float64Array(Nx);
  for (let i = 0; i < Nx; i++) xgrid[i] = -L + i * dx;

  const tgt = { mu: [-3, 3], s: 0.5, w: [0.5, 0.5] };  // symmetric bimodal target

  // ---- state ----
  let u = new Float64Array(Nx * Nt);
  let iter = 0, eta = 10, playing = false, timer = null;
  let lastX1 = null, lastTrajs = [];

  // ---- rng ----
  function gauss() {
    let a = 0, b = 0;
    while (a === 0) a = Math.random();
    while (b === 0) b = Math.random();
    return Math.sqrt(-2 * Math.log(a)) * Math.cos(2 * Math.PI * b);
  }

  // ---- target / prior densities & scores ----
  function tScore(x) {
    const { mu, s, w } = tgt; let num = 0, den = 0;
    for (let k = 0; k < mu.length; k++) {
      const e = w[k] * Math.exp(-0.5 * ((x - mu[k]) / s) ** 2);
      den += e; num += e * (-(x - mu[k]) / (s * s));
    }
    return num / (den + 1e-12);
  }
  function tPdf(x) {
    const { mu, s, w } = tgt; let p = 0;
    const c = 1 / (s * Math.sqrt(2 * Math.PI));
    for (let k = 0; k < mu.length; k++) p += w[k] * c * Math.exp(-0.5 * ((x - mu[k]) / s) ** 2);
    return p;
  }
  function pPdf(x) {
    return Math.exp(-0.5 * (x / sigma_p) ** 2) / (sigma_p * Math.sqrt(2 * Math.PI));
  }

  // ---- control interpolation (linear in x at column j) ----
  function interpU(x, j) {
    let f = (x + L) / dx;
    if (f <= 0) return u[0 * Nt + j];
    if (f >= Nx - 1) return u[(Nx - 1) * Nt + j];
    const i = f | 0, frac = f - i;
    return u[i * Nt + j] * (1 - frac) + u[(i + 1) * Nt + j] * frac;
  }

  // ---- simulate the controlled SDE (Euler-Maruyama, drift = g^2 u) ----
  function simulate(captureK) {
    const X1 = new Float64Array(B);
    const trajs = [];
    const sdt = g * Math.sqrt(dt);
    for (let b = 0; b < B; b++) {
      let x = sigma_p * gauss();
      const keep = b < captureK; const path = keep ? [x] : null;
      for (let j = 0; j < Nt - 1; j++) {
        x += g * g * interpU(x, j) * dt + sdt * gauss();
        if (keep) path.push(x);
      }
      X1[b] = x;
      if (keep) trajs.push(path);
    }
    return { X1, trajs };
  }

  // ---- Markovianize: u_new(x,t) = E[nelson | X_t = x] via windowed NW ----
  function markovianize(X1) {
    const X0p = new Float64Array(B), ps = new Float64Array(B),
          ts = new Float64Array(B), base = new Float64Array(B);
    for (let b = 0; b < B; b++) {
      const xp = sigma_p * gauss();
      X0p[b] = xp; ps[b] = -xp / (sigma_p * sigma_p);
      ts[b] = tScore(X1[b]); base[b] = (X1[b] - xp) / kappa1;  // -ref_score_1_0
    }
    const num = new Float64Array(Nx * Nt), den = new Float64Array(Nx * Nt);
    for (let j = 0; j < Nt; j++) {
      const t = j * dt, sd = Math.sqrt(g * g * t * (1 - t) + 1e-9);
      for (let b = 0; b < B; b++) {
        const Xt = (1 - t) * X0p[b] + t * X1[b] + sd * gauss();
        const nelson = t * (ps[b] + ts[b]) + base[b];
        let ic = Math.round((Xt + L) / dx);
        const lo = Math.max(0, ic - halfW), hi = Math.min(Nx - 1, ic + halfW);
        for (let i = lo; i <= hi; i++) {
          const d = (xgrid[i] - Xt) / h, wk = Math.exp(-0.5 * d * d);
          num[i * Nt + j] += wk * nelson; den[i * Nt + j] += wk;
        }
      }
    }
    return { num, den };
  }

  // ---- one outer fixed-point iteration (damped) ----
  function step() {
    const { X1, trajs } = simulate(14);
    const { num, den } = markovianize(X1);
    const alpha = 1 / (1 + eta);
    for (let k = 0; k < Nx * Nt; k++) {
      if (den[k] > 1e-3) {
        const uNew = num[k] / den[k];
        u[k] = alpha * uNew + (1 - alpha) * u[k];
      }
    }
    iter++; lastX1 = X1; lastTrajs = trajs;
    document.getElementById('iterVal').textContent = iter;
    draw();
  }

  function reset() {
    u = new Float64Array(Nx * Nt); iter = 0;
    const { X1, trajs } = simulate(14);
    lastX1 = X1; lastTrajs = trajs;
    document.getElementById('iterVal').textContent = 0;
    draw();
  }

  // ---- color map (diverging blue-white-red) ----
  function divColor(v, vmax) {
    const t = Math.max(-1, Math.min(1, v / vmax));
    let r, g_, b;
    if (t >= 0) { r = 255; g_ = 255 - t * 150; b = 255 - t * 190; }
    else { const a = -t; r = 255 - a * 200; g_ = 255 - a * 120; b = 255; }
    return `rgb(${r | 0},${g_ | 0},${b | 0})`;
  }

  // ---- draw field + trajectories ----
  function drawField() {
    const W = field.width, H = field.height;
    const pL = 38, pR = 8, pT = 8, pB = 22;
    fx.clearRect(0, 0, W, H);
    let vmax = 0;
    for (let k = 0; k < u.length; k++) vmax = Math.max(vmax, Math.abs(u[k]));
    vmax = Math.max(2, Math.min(10, vmax));
    const cw = (W - pL - pR) / (Nt - 1), ch = (H - pT - pB) / Nx;
    for (let j = 0; j < Nt; j++) {
      for (let i = 0; i < Nx; i++) {
        fx.fillStyle = divColor(u[(Nx - 1 - i) * Nt + j], vmax);
        fx.fillRect(pL + (j - 0.5) * cw, pT + i * ch, cw + 1, ch + 1);
      }
    }
    // trajectories
    const tx = t => pL + t * (W - pL - pR);
    const xy = x => pT + (L - x) / (2 * L) * (H - pT - pB);
    fx.strokeStyle = 'rgba(30,40,60,0.5)'; fx.lineWidth = 1;
    for (const path of lastTrajs) {
      fx.beginPath();
      for (let j = 0; j < path.length; j++) {
        const px = tx(j * dt), py = xy(Math.max(-L, Math.min(L, path[j])));
        j === 0 ? fx.moveTo(px, py) : fx.lineTo(px, py);
      }
      fx.stroke();
    }
    // frame + labels
    fx.strokeStyle = '#d7dbe4'; fx.strokeRect(pL, pT, W - pL - pR, H - pT - pB);
    fx.fillStyle = '#8b93a4'; fx.font = '11px Inter, sans-serif';
    fx.textAlign = 'center';
    fx.fillText('t=0', pL, H - 7); fx.fillText('t=1', W - pR, H - 7);
    fx.save(); fx.translate(11, H / 2); fx.rotate(-Math.PI / 2);
    fx.fillText('x', 0, 0); fx.restore();
  }

  // ---- draw terminal histogram vs target ----
  function drawHist() {
    const W = hist.width, H = hist.height;
    const pL = 34, pR = 8, pT = 10, pB = 22;
    hx.clearRect(0, 0, W, H);
    const nb = 56, binw = 2 * L / nb;
    const counts = new Float64Array(nb);
    if (lastX1) for (let b = 0; b < lastX1.length; b++) {
      let bi = Math.floor((lastX1[b] + L) / binw);
      if (bi >= 0 && bi < nb) counts[bi]++;
    }
    const dens = counts.map(c => c / (B * binw));
    let ymax = 0;
    for (const d of dens) ymax = Math.max(ymax, d);
    for (let i = 0; i <= 200; i++) ymax = Math.max(ymax, tPdf(-L + i / 200 * 2 * L));
    ymax *= 1.15;

    const tx = x => pL + (x + L) / (2 * L) * (W - pL - pR);
    const ty = d => H - pB - d / ymax * (H - pT - pB);

    // bars
    hx.fillStyle = 'rgba(142,68,173,0.45)';
    for (let i = 0; i < nb; i++) {
      const x0 = tx(-L + i * binw), x1 = tx(-L + (i + 1) * binw);
      hx.fillRect(x0, ty(dens[i]), x1 - x0 - 0.5, H - pB - ty(dens[i]));
    }
    // prior (faint blue) + target (red)
    const curve = (fn, color, wdt, alpha) => {
      hx.strokeStyle = color; hx.lineWidth = wdt; hx.globalAlpha = alpha;
      hx.beginPath();
      for (let i = 0; i <= 240; i++) {
        const x = -L + i / 240 * 2 * L, px = tx(x), py = ty(fn(x));
        i === 0 ? hx.moveTo(px, py) : hx.lineTo(px, py);
      }
      hx.stroke(); hx.globalAlpha = 1;
    };
    curve(pPdf, '#2c6fbb', 1.5, 0.5);
    curve(tPdf, '#c0392b', 2.2, 1);

    // frame + labels
    hx.strokeStyle = '#d7dbe4'; hx.strokeRect(pL, pT, W - pL - pR, H - pT - pB);
    hx.fillStyle = '#8b93a4'; hx.font = '11px Inter, sans-serif'; hx.textAlign = 'center';
    for (const xv of [-3, 0, 3]) hx.fillText(String(xv), tx(xv), H - 7);
    hx.fillText('Xₜ', (pL + W - pR) / 2, H - 7);
  }

  function draw() { drawField(); drawHist(); }

  // ---- controls ----
  const stepBtn = document.getElementById('stepBtn');
  const playBtn = document.getElementById('playBtn');
  const resetBtn = document.getElementById('resetBtn');
  const etaSlider = document.getElementById('etaSlider');
  const etaVal = document.getElementById('etaVal');
  const alphaVal = document.getElementById('alphaVal');

  function setPlaying(on) {
    playing = on; playBtn.textContent = on ? 'Pause ❚❚' : 'Play ▶';
    if (on) { timer = setInterval(step, 220); } else { clearInterval(timer); }
  }
  stepBtn.addEventListener('click', () => { if (!playing) step(); });
  playBtn.addEventListener('click', () => setPlaying(!playing));
  resetBtn.addEventListener('click', () => { reset(); });
  etaSlider.addEventListener('input', () => {
    eta = +etaSlider.value; etaVal.textContent = eta.toFixed(eta % 1 ? 1 : 0);
    alphaVal.textContent = (1 / (1 + eta)).toFixed(2);
  });

  // Warm up so the demo looks alive on first view; Reset returns to iteration 0.
  reset();
  for (let i = 0; i < 12; i++) step();
})();
