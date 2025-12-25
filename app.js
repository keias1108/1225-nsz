/**
 * @fileoverview
 * 브라우저(Canvas)에서 A/B 벡터장, z(스케일/시간축), u(발산 위험도)를 시뮬레이션·시각화하는 단일 페이지 로직.
 * 의존성: 표준 Web API(Canvas 2D, DOM)만 사용.
 */

const EPS = 1e-8;

function clamp(value, min, max) {
  return Math.max(min, Math.min(max, value));
}

function sigmoid(x) {
  // 수치 안정성을 위해 클램프
  const t = clamp(x, -60, 60);
  return 1 / (1 + Math.exp(-t));
}

function rotate90(x, y) {
  return [-y, x];
}

function hsvToRgb(h, s, v) {
  const c = v * s;
  const hp = ((h % 360) + 360) % 360 / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r = 0,
    g = 0,
    b = 0;
  if (hp >= 0 && hp < 1) [r, g, b] = [c, x, 0];
  else if (hp < 2) [r, g, b] = [x, c, 0];
  else if (hp < 3) [r, g, b] = [0, c, x];
  else if (hp < 4) [r, g, b] = [0, x, c];
  else if (hp < 5) [r, g, b] = [x, 0, c];
  else [r, g, b] = [c, 0, x];
  const m = v - c;
  return [Math.round((r + m) * 255), Math.round((g + m) * 255), Math.round((b + m) * 255)];
}

function colorMap(value, min, max) {
  if (!Number.isFinite(value)) return [255, 0, 255];
  const t = max > min ? (value - min) / (max - min) : 0.5;
  const tt = clamp(t, 0, 1);
  // 파랑(220) -> 빨강(10)
  const h = 220 + (10 - 220) * tt;
  const s = 0.95;
  const v = 0.95;
  return hsvToRgb(h, s, v);
}

class Simulator {
  constructor(canvas, statsEl) {
    this.canvas = canvas;
    this.ctx = canvas.getContext("2d", { alpha: false });
    this.statsEl = statsEl;

    this.w = canvas.width;
    this.h = canvas.height;
    this.n = this.w * this.h;

    this.aX = new Float32Array(this.n);
    this.aY = new Float32Array(this.n);
    this.bX = new Float32Array(this.n);
    this.bY = new Float32Array(this.n);
    this.z = new Float32Array(this.n);

    this.u = new Float32Array(this.n);
    this.s = new Float32Array(this.n);
    this.m = new Float32Array(this.n);
    this.g = new Float32Array(this.n);
    this.dot = new Float32Array(this.n);
    this.aMag = new Float32Array(this.n);
    this.bMag = new Float32Array(this.n);

    this.tmpAX = new Float32Array(this.n);
    this.tmpAY = new Float32Array(this.n);
    this.tmpBX = new Float32Array(this.n);
    this.tmpBY = new Float32Array(this.n);
    this.tmpZ = new Float32Array(this.n);

    this.imageData = this.ctx.createImageData(this.w, this.h);
    this.running = true;
    this.viewField = "u";
    this.showVectors = true;
    this.mode = "timeSlow";

    this.params = {
      stepsPerFrame: 4,
      dt: 0.2,
      diff: 0.25,
      noise: 0.02,
      maxNorm: 8,
      growthK: 1.1,
      linearDamp: 0.35,
      nonlinearDamp: 0.03,
      ku: 2.2,
      lambdaZ: 0.8,
      beta: 1.6,
      eta: 2.5,
      thetaS: 0.8,
      thetaM: 0.85,
      gc: 0.35,
      alphaS: 8,
      alphaM: 10,
      alphaG: 8,
      brushRadius: 14,
      brushStrength: 1.2,
    };

    this.t = 0;
    this.frame = 0;
    this.lastFpsTime = performance.now();
    this.fps = 0;

    this.reset();
  }

  idx(x, y) {
    return y * this.w + x;
  }

  wrapX(x) {
    const ww = this.w;
    return (x + ww) % ww;
  }

  wrapY(y) {
    const hh = this.h;
    return (y + hh) % hh;
  }

  reset() {
    for (let i = 0; i < this.n; i++) {
      this.aX[i] = (Math.random() - 0.5) * 0.25;
      this.aY[i] = (Math.random() - 0.5) * 0.25;
      this.bX[i] = (Math.random() - 0.5) * 0.25;
      this.bY[i] = (Math.random() - 0.5) * 0.25;
      this.z[i] = 0;
    }
    // 중앙에 작은 시드(정렬 소용돌이)
    this.paintVortex(Math.floor(this.w * 0.5), Math.floor(this.h * 0.5), 18, 0.9, "aligned");
    this.t = 0;
  }

  randomize() {
    for (let i = 0; i < this.n; i++) {
      this.aX[i] = (Math.random() - 0.5) * 1.2;
      this.aY[i] = (Math.random() - 0.5) * 1.2;
      this.bX[i] = (Math.random() - 0.5) * 1.2;
      this.bY[i] = (Math.random() - 0.5) * 1.2;
      this.z[i] = Math.random() * 0.2;
    }
    this.t = 0;
  }

  computeLocalScalars() {
    const { growthK, thetaS, thetaM, gc, alphaS, alphaM, alphaG } = this.params;
    for (let i = 0; i < this.n; i++) {
      const ax = this.aX[i];
      const ay = this.aY[i];
      const bx = this.bX[i];
      const by = this.bY[i];
      const dot = ax * bx + ay * by;
      const nA = Math.hypot(ax, ay);
      const nB = Math.hypot(bx, by);

      const s = dot / (nA * nB + EPS); // [-1,1]
      const m = Math.min(nA, nB) / (Math.max(nA, nB) + EPS); // [0,1]
      const g = growthK * dot;

      const u =
        sigmoid(alphaS * (s - thetaS)) * sigmoid(alphaM * (m - thetaM)) * sigmoid(alphaG * (g - gc));

      this.dot[i] = dot;
      this.aMag[i] = nA;
      this.bMag[i] = nB;
      this.s[i] = s;
      this.m[i] = m;
      this.g[i] = g;
      this.u[i] = u;
    }
  }

  stepOnce() {
    const {
      dt,
      diff,
      noise,
      maxNorm,
      growthK,
      linearDamp,
      nonlinearDamp,
      ku,
      lambdaZ,
      beta,
      eta,
    } = this.params;

    this.computeLocalScalars();

    // z 업데이트 + (옵션 A) Δt_eff 결정
    for (let i = 0; i < this.n; i++) {
      const u = this.u[i];
      const zNext = this.z[i] + dt * (ku * u - lambdaZ * this.z[i]);
      this.tmpZ[i] = clamp(zNext, 0, 50);
    }

    const useTimeSlow = this.mode === "timeSlow" || this.mode === "both";
    const useKernelScale = this.mode === "kernelScale" || this.mode === "both";

    for (let y = 0; y < this.h; y++) {
      const yU = this.wrapY(y - 1);
      const yD = this.wrapY(y + 1);
      for (let x = 0; x < this.w; x++) {
        const xL = this.wrapX(x - 1);
        const xR = this.wrapX(x + 1);
        const i = this.idx(x, y);
        const iL = this.idx(xL, y);
        const iR = this.idx(xR, y);
        const iU = this.idx(x, yU);
        const iD = this.idx(x, yD);

        const ax = this.aX[i];
        const ay = this.aY[i];
        const bx = this.bX[i];
        const by = this.bY[i];

        const dot = this.dot[i];
        const nA = this.aMag[i];
        const nB = this.bMag[i];
        const u = this.u[i];
        const z = this.tmpZ[i];

        const dtEff = useTimeSlow ? dt * Math.exp(-beta * z) : dt;
        const diffEff = useKernelScale ? diff * Math.exp(clamp(beta * z, 0, 4.0)) : diff;

        const lapAX = this.aX[iL] + this.aX[iR] + this.aX[iU] + this.aX[iD] - 4 * ax;
        const lapAY = this.aY[iL] + this.aY[iR] + this.aY[iU] + this.aY[iD] - 4 * ay;
        const lapBX = this.bX[iL] + this.bX[iR] + this.bX[iU] + this.bX[iD] - 4 * bx;
        const lapBY = this.bY[iL] + this.bY[iR] + this.bY[iU] + this.bY[iD] - 4 * by;

        const inter = growthK * dot; // 정렬(양의 dot)일수록 성장

        const nA2 = nA * nA;
        const nB2 = nB * nB;

        const [rAX, rAY] = rotate90(ax, ay);
        const [rBX, rBY] = rotate90(bx, by);

        const noiseAX = (Math.random() - 0.5) * noise;
        const noiseAY = (Math.random() - 0.5) * noise;
        const noiseBX = (Math.random() - 0.5) * noise;
        const noiseBY = (Math.random() - 0.5) * noise;

        let dAX =
          diffEff * lapAX +
          inter * ax -
          linearDamp * ax -
          nonlinearDamp * nA2 * ax +
          eta * u * rAX +
          noiseAX;
        let dAY =
          diffEff * lapAY +
          inter * ay -
          linearDamp * ay -
          nonlinearDamp * nA2 * ay +
          eta * u * rAY +
          noiseAY;

        let dBX =
          diffEff * lapBX +
          inter * bx -
          linearDamp * bx -
          nonlinearDamp * nB2 * bx -
          eta * u * rBX +
          noiseBX;
        let dBY =
          diffEff * lapBY +
          inter * by -
          linearDamp * by -
          nonlinearDamp * nB2 * by -
          eta * u * rBY +
          noiseBY;

        let axN = ax + dtEff * dAX;
        let ayN = ay + dtEff * dAY;
        let bxN = bx + dtEff * dBX;
        let byN = by + dtEff * dBY;

        // NaN/무한 방지용 클램프(표현 가능한 "폭주 직전"만 유지)
        const aNormN = Math.hypot(axN, ayN);
        if (aNormN > maxNorm) {
          const k = maxNorm / (aNormN + EPS);
          axN *= k;
          ayN *= k;
        }
        const bNormN = Math.hypot(bxN, byN);
        if (bNormN > maxNorm) {
          const k = maxNorm / (bNormN + EPS);
          bxN *= k;
          byN *= k;
        }

        this.tmpAX[i] = axN;
        this.tmpAY[i] = ayN;
        this.tmpBX[i] = bxN;
        this.tmpBY[i] = byN;
      }
    }

    // swap
    [this.aX, this.tmpAX] = [this.tmpAX, this.aX];
    [this.aY, this.tmpAY] = [this.tmpAY, this.aY];
    [this.bX, this.tmpBX] = [this.tmpBX, this.bX];
    [this.bY, this.tmpBY] = [this.tmpBY, this.bY];
    [this.z, this.tmpZ] = [this.tmpZ, this.z];
    this.t += dt;
  }

  paintVortex(cx, cy, radius, strength, kind) {
    const r2 = radius * radius;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const d2 = dx * dx + dy * dy;
        if (d2 > r2) continue;
        const x = this.wrapX(cx + dx);
        const y = this.wrapY(cy + dy);
        const i = this.idx(x, y);

        const falloff = 1 - Math.sqrt(d2) / (radius + EPS);
        const px = dx / (Math.sqrt(d2) + EPS);
        const py = dy / (Math.sqrt(d2) + EPS);
        // 접선 방향(소용돌이)
        const tx = -py;
        const ty = px;
        const s = strength * falloff;

        this.aX[i] += tx * s;
        this.aY[i] += ty * s;
        if (kind === "aligned") {
          this.bX[i] += tx * s;
          this.bY[i] += ty * s;
        } else if (kind === "opposite") {
          this.bX[i] -= tx * s;
          this.bY[i] -= ty * s;
        }
      }
    }
  }

  paintAlignedLine(cx, cy, radius, strength) {
    const r2 = radius * radius;
    const dirX = 1;
    const dirY = 0;
    for (let dy = -radius; dy <= radius; dy++) {
      for (let dx = -radius; dx <= radius; dx++) {
        const d2 = dx * dx + dy * dy;
        if (d2 > r2) continue;
        const x = this.wrapX(cx + dx);
        const y = this.wrapY(cy + dy);
        const i = this.idx(x, y);
        const falloff = 1 - Math.sqrt(d2) / (radius + EPS);
        const s = strength * falloff;
        this.aX[i] += dirX * s;
        this.aY[i] += dirY * s;
        this.bX[i] += dirX * s;
        this.bY[i] += dirY * s;
      }
    }
  }

  render() {
    this.computeLocalScalars();

    const field = this[this.viewField] ?? this.u;
    let min = Infinity;
    let max = -Infinity;
    for (let i = 0; i < this.n; i++) {
      const v = field[i];
      if (v < min) min = v;
      if (v > max) max = v;
    }
    // 보기 좋게 특정 필드는 범위 고정
    if (this.viewField === "u" || this.viewField === "m") {
      min = 0;
      max = 1;
    } else if (this.viewField === "s") {
      min = -1;
      max = 1;
    } else if (this.viewField === "z") {
      min = 0;
      max = Math.max(1, max);
    }

    const data = this.imageData.data;
    for (let i = 0; i < this.n; i++) {
      const [r, g, b] = colorMap(field[i], min, max);
      const o = i * 4;
      data[o + 0] = r;
      data[o + 1] = g;
      data[o + 2] = b;
      data[o + 3] = 255;
    }
    this.ctx.putImageData(this.imageData, 0, 0);

    if (this.showVectors) this.drawVectors();

    this.updateStats(min, max);
  }

  drawVectors() {
    const ctx = this.ctx;
    ctx.save();
    ctx.globalAlpha = 0.8;
    ctx.lineWidth = 1;

    const step = Math.max(6, Math.floor(Math.min(this.w, this.h) / 22));
    for (let y = 0; y < this.h; y += step) {
      for (let x = 0; x < this.w; x += step) {
        const i = this.idx(x, y);
        const ax = this.aX[i];
        const ay = this.aY[i];
        const bx = this.bX[i];
        const by = this.bY[i];
        const u = this.u[i];

        const aN = Math.hypot(ax, ay);
        const bN = Math.hypot(bx, by);

        const lenA = clamp(aN * 2.2, 0, step * 0.55);
        const lenB = clamp(bN * 2.2, 0, step * 0.55);

        // u가 높을수록 밝게
        const alpha = 0.15 + 0.75 * u;

        // A: 시안
        ctx.strokeStyle = `rgba(120, 230, 255, ${alpha})`;
        if (lenA > 0.5) {
          ctx.beginPath();
          ctx.moveTo(x + 0.5, y + 0.5);
          ctx.lineTo(x + 0.5 + (ax / (aN + EPS)) * lenA, y + 0.5 + (ay / (aN + EPS)) * lenA);
          ctx.stroke();
        }

        // B: 핑크
        ctx.strokeStyle = `rgba(255, 140, 210, ${alpha})`;
        if (lenB > 0.5) {
          ctx.beginPath();
          ctx.moveTo(x + 0.5, y + 0.5);
          ctx.lineTo(x + 0.5 + (bx / (bN + EPS)) * lenB, y + 0.5 + (by / (bN + EPS)) * lenB);
          ctx.stroke();
        }
      }
    }
    ctx.restore();
  }

  updateStats(viewMin, viewMax) {
    let maxU = 0;
    let maxZ = 0;
    let maxAMag = 0;
    let maxBMag = 0;
    let meanU = 0;
    for (let i = 0; i < this.n; i++) {
      const u = this.u[i];
      const z = this.z[i];
      const a = this.aMag[i];
      const b = this.bMag[i];
      if (u > maxU) maxU = u;
      if (z > maxZ) maxZ = z;
      if (a > maxAMag) maxAMag = a;
      if (b > maxBMag) maxBMag = b;
      meanU += u;
    }
    meanU /= this.n;

    const now = performance.now();
    const dt = now - this.lastFpsTime;
    if (dt > 500) {
      this.fps = Math.round((1000 * (this.frame + 1)) / dt);
      this.frame = 0;
      this.lastFpsTime = now;
    } else {
      this.frame++;
    }

    const p = this.params;
    this.statsEl.textContent =
      `FPS: ${this.fps}\n` +
      `t: ${this.t.toFixed(2)}   모드: ${this.mode}\n` +
      `보기(${this.viewField}): [${viewMin.toFixed(3)}, ${viewMax.toFixed(3)}]\n` +
      `u: mean=${meanU.toFixed(3)} max=${maxU.toFixed(3)}\n` +
      `z: max=${maxZ.toFixed(3)}   |A|max=${maxAMag.toFixed(3)}   |B|max=${maxBMag.toFixed(3)}\n` +
      `β=${p.beta.toFixed(2)} η=${p.eta.toFixed(2)} k_u=${p.ku.toFixed(2)} λ=${p.lambdaZ.toFixed(2)}\n` +
      `θs=${p.thetaS.toFixed(2)} θm=${p.thetaM.toFixed(2)} gc=${p.gc.toFixed(2)} (αs=${p.alphaS.toFixed(
        1
      )}, αm=${p.alphaM.toFixed(1)}, αg=${p.alphaG.toFixed(1)})`;
  }
}

function setupControls(sim) {
  const panelControls = [...document.querySelectorAll(".control")];
  for (const el of panelControls) {
    const key = el.dataset.key;
    if (!key) continue;
    const min = Number(el.dataset.min);
    const max = Number(el.dataset.max);
    const step = Number(el.dataset.step);
    const value = Number(el.dataset.value);
    if (Number.isFinite(value)) sim.params[key] = value;

    const line = document.createElement("div");
    line.className = "line";
    const input = document.createElement("input");
    input.type = "range";
    input.min = String(min);
    input.max = String(max);
    input.step = String(step);
    input.value = String(sim.params[key]);
    const out = document.createElement("output");
    out.value = String(sim.params[key]);
    line.appendChild(input);
    line.appendChild(out);
    el.appendChild(line);

    const update = () => {
      sim.params[key] = Number(input.value);
      out.value = Number(input.value).toString();
    };
    input.addEventListener("input", update);
    update();
  }

  const btnToggle = document.getElementById("btnToggle");
  const btnStep = document.getElementById("btnStep");
  const btnReset = document.getElementById("btnReset");
  const btnRandomize = document.getElementById("btnRandomize");

  btnToggle.addEventListener("click", () => {
    sim.running = !sim.running;
    btnToggle.textContent = sim.running ? "일시정지" : "재생";
  });
  btnStep.addEventListener("click", () => sim.stepOnce());
  btnReset.addEventListener("click", () => sim.reset());
  btnRandomize.addEventListener("click", () => sim.randomize());

  const viewField = document.getElementById("viewField");
  viewField.addEventListener("change", () => (sim.viewField = viewField.value));
  sim.viewField = viewField.value;

  const showVectors = document.getElementById("showVectors");
  showVectors.addEventListener("change", () => (sim.showVectors = showVectors.checked));
  sim.showVectors = showVectors.checked;

  const mode = document.getElementById("mode");
  mode.addEventListener("change", () => (sim.mode = mode.value));
  sim.mode = mode.value;

  const brushType = document.getElementById("brushType");
  let currentBrushType = brushType.value;
  brushType.addEventListener("change", () => (currentBrushType = brushType.value));

  // 드래그 주입
  const canvas = sim.canvas;
  let dragging = false;
  const paintAt = (ev) => {
    const rect = canvas.getBoundingClientRect();
    const x = Math.floor(((ev.clientX - rect.left) / rect.width) * sim.w);
    const y = Math.floor(((ev.clientY - rect.top) / rect.height) * sim.h);
    const r = Math.floor(sim.params.brushRadius);
    const s = sim.params.brushStrength;

    if (currentBrushType === "alignedVortex") sim.paintVortex(x, y, r, s, "aligned");
    else if (currentBrushType === "oppositeVortex") sim.paintVortex(x, y, r, s, "opposite");
    else sim.paintAlignedLine(x, y, r, s);
  };

  canvas.addEventListener("pointerdown", (ev) => {
    dragging = true;
    canvas.setPointerCapture(ev.pointerId);
    paintAt(ev);
  });
  canvas.addEventListener("pointermove", (ev) => {
    if (!dragging) return;
    paintAt(ev);
  });
  canvas.addEventListener("pointerup", () => (dragging = false));
  canvas.addEventListener("pointercancel", () => (dragging = false));
}

function main() {
  const canvas = document.getElementById("sim");
  const statsEl = document.getElementById("stats");
  const sim = new Simulator(canvas, statsEl);
  setupControls(sim);

  const loop = () => {
    if (sim.running) {
      const k = Math.max(1, Math.floor(sim.params.stepsPerFrame));
      for (let i = 0; i < k; i++) sim.stepOnce();
    }
    sim.render();
    requestAnimationFrame(loop);
  };
  requestAnimationFrame(loop);
}

window.addEventListener("DOMContentLoaded", main);

