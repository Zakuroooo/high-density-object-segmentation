/*
 * yolo.js — YOLOv8-seg pre/post-processing in the browser.
 *
 * The model exports two tensors and nothing else; everything that turns them
 * into instance masks normally happens inside Ultralytics' Python. Running the
 * demo without a server means reimplementing that here:
 *
 *   output0  (1, 116, 8400)   4 box + 80 class scores + 32 mask coefficients,
 *                             per anchor, across strides 8/16/32 at 640px
 *                             (80² + 40² + 20² = 8400)
 *   output1  (1, 32, 160, 160) 32 mask prototypes
 *
 * A mask is the sigmoid of the prototypes combined by an instance's 32
 * coefficients, cropped to that instance's box. That linear-combination trick
 * is what lets one head emit a variable number of instance masks.
 */

export const INPUT = 640;
export const PROTO = 160;
export const NUM_CLASSES = 80;
export const NUM_COEFF = 32;

/* Letterbox: resize preserving aspect ratio, pad the remainder with grey.
 * Padding with grey rather than black matters — black is a colour the network
 * has learned to read as content, and the padding would leak into predictions
 * near the image edge. 114 is the value Ultralytics trains with. */
export function letterbox(img) {
  const c = document.createElement("canvas");
  c.width = c.height = INPUT;
  const ctx = c.getContext("2d", { willReadFrequently: true });
  ctx.fillStyle = "rgb(114,114,114)";
  ctx.fillRect(0, 0, INPUT, INPUT);

  const scale = Math.min(INPUT / img.width, INPUT / img.height);
  const w = Math.round(img.width * scale);
  const h = Math.round(img.height * scale);
  const dx = Math.floor((INPUT - w) / 2);
  const dy = Math.floor((INPUT - h) / 2);
  ctx.drawImage(img, dx, dy, w, h);

  return { canvas: c, ctx, scale, dx, dy, w, h };
}

/* HWC uint8 RGBA -> NCHW float32 RGB in [0,1], the layout the graph expects. */
export function toTensor(ctx) {
  const { data } = ctx.getImageData(0, 0, INPUT, INPUT);
  const n = INPUT * INPUT;
  const out = new Float32Array(3 * n);
  for (let i = 0; i < n; i++) {
    out[i] = data[i * 4] / 255;             // R plane
    out[n + i] = data[i * 4 + 1] / 255;     // G plane
    out[2 * n + i] = data[i * 4 + 2] / 255; // B plane
  }
  return out;
}

const sigmoid = (x) => 1 / (1 + Math.exp(-x));

function iou(a, b) {
  const x1 = Math.max(a.x1, b.x1), y1 = Math.max(a.y1, b.y1);
  const x2 = Math.min(a.x2, b.x2), y2 = Math.min(a.y2, b.y2);
  const inter = Math.max(0, x2 - x1) * Math.max(0, y2 - y1);
  if (inter <= 0) return 0;
  const areaA = (a.x2 - a.x1) * (a.y2 - a.y1);
  const areaB = (b.x2 - b.x1) * (b.y2 - b.y1);
  return inter / (areaA + areaB - inter);
}

/* Per-class NMS, matching Ultralytics' default (agnostic=False): boxes only
 * suppress boxes of the same class, so a person standing in front of a car
 * does not delete the car. */
function nms(dets, iouThr) {
  dets.sort((p, q) => q.score - p.score);
  const keep = [];
  const dead = new Uint8Array(dets.length);
  for (let i = 0; i < dets.length; i++) {
    if (dead[i]) continue;
    keep.push(dets[i]);
    for (let j = i + 1; j < dets.length; j++) {
      if (dead[j] || dets[j].cls !== dets[i].cls) continue;
      if (iou(dets[i], dets[j]) > iouThr) dead[j] = 1;
    }
  }
  return keep;
}

/* Decode output0. It arrives transposed — feature-major (116 rows of 8400)
 * rather than anchor-major — so every read strides by 8400. */
export function decode(out0, conf, iouThr, maxDet = 300) {
  const A = 8400;
  const dets = [];
  for (let a = 0; a < A; a++) {
    let best = -1, bestCls = -1;
    for (let c = 0; c < NUM_CLASSES; c++) {
      const s = out0[(4 + c) * A + a];
      if (s > best) { best = s; bestCls = c; }
    }
    if (best < conf) continue;

    const cx = out0[0 * A + a], cy = out0[1 * A + a];
    const w = out0[2 * A + a], h = out0[3 * A + a];
    const coeff = new Float32Array(NUM_COEFF);
    for (let k = 0; k < NUM_COEFF; k++) {
      coeff[k] = out0[(4 + NUM_CLASSES + k) * A + a];
    }
    dets.push({
      x1: cx - w / 2, y1: cy - h / 2, x2: cx + w / 2, y2: cy + h / 2,
      score: best, cls: bestCls, coeff,
    });
  }
  return nms(dets, iouThr).slice(0, maxDet);
}

/* Build one instance mask: sigmoid(coeff · protos), then crop to the box.
 *
 * The crop is not cosmetic. Prototypes are global, so a coefficient vector
 * that lights up "person-shaped blobs" activates on every person in the frame;
 * without the box crop, one detection's mask would cover all of them. */
export function buildMask(det, protos, lb, outW, outH) {
  const P = PROTO;
  const raw = new Float32Array(P * P);
  for (let k = 0; k < NUM_COEFF; k++) {
    const c = det.coeff[k];
    if (c === 0) continue;
    const base = k * P * P;
    for (let i = 0; i < P * P; i++) raw[i] += c * protos[base + i];
  }

  // Box in 640-space -> prototype space (a factor of 4).
  const r = P / INPUT;
  const bx1 = Math.max(0, Math.floor(det.x1 * r));
  const by1 = Math.max(0, Math.floor(det.y1 * r));
  const bx2 = Math.min(P, Math.ceil(det.x2 * r));
  const by2 = Math.min(P, Math.ceil(det.y2 * r));

  // Sample straight from prototype space into original-image space, undoing
  // the letterbox as we go — one resample instead of three.
  const mask = new Uint8Array(outW * outH);
  for (let y = 0; y < outH; y++) {
    const yl = y * lb.scale + lb.dy;      // original -> letterboxed 640
    const yp = yl * r;                    // -> prototype space
    const ypi = Math.round(yp);
    if (ypi < by1 || ypi >= by2) continue;
    for (let x = 0; x < outW; x++) {
      const xl = x * lb.scale + lb.dx;
      const xpi = Math.round(xl * r);
      if (xpi < bx1 || xpi >= bx2) continue;
      if (sigmoid(raw[ypi * P + xpi]) > 0.5) mask[y * outW + x] = 1;
    }
  }
  return mask;
}

/* ── Canny edge density — the hybrid's gate ──────────────────────────────────
 *
 * Reimplemented rather than approximated with a plain Sobel magnitude, because
 * the 0.08 threshold in the paper was measured against OpenCV's Canny(50,150).
 * A different edge operator would fire at a different rate and the demo would
 * not be showing the method that was evaluated. Same five stages OpenCV uses. */
export function cannyEdgeDensity(ctx, w, h, lo = 50, hi = 150) {
  const { data } = ctx.getImageData(0, 0, w, h);
  const n = w * h;

  const blur = new Float32Array(n);
  for (let i = 0; i < n; i++) {
    blur[i] = 0.299 * data[i * 4] + 0.587 * data[i * 4 + 1] + 0.114 * data[i * 4 + 2];
  }

  // NOTE: no Gaussian pre-blur, deliberately.
  //
  // Textbook Canny begins with a Gaussian, and an earlier version of this
  // function did too — which made the demo disagree with the paper. OpenCV's
  // cv2.Canny() does NOT blur internally; it expects the caller to have done it,
  // and src/hybrid.py calls cv2.Canny(gray, 50, 150) on the raw grayscale. The
  // blur was removing enough weak edges to report 0.1084 where OpenCV reported
  // 0.2622 on the same image — a factor of 2.4, straddling the 0.08 gate
  // threshold that the published results were calibrated against.

  // Sobel gradients.
  const mag = new Float32Array(n), dir = new Uint8Array(n);
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      const gx =
        -blur[i - w - 1] + blur[i - w + 1] +
        -2 * blur[i - 1] + 2 * blur[i + 1] +
        -blur[i + w - 1] + blur[i + w + 1];
      const gy =
        -blur[i - w - 1] - 2 * blur[i - w] - blur[i - w + 1] +
        blur[i + w - 1] + 2 * blur[i + w] + blur[i + w + 1];
      // L1 magnitude, |gx| + |gy|. OpenCV's Canny defaults to L2gradient=False,
      // i.e. this approximation rather than sqrt(gx² + gy²). Using the true L2
      // norm here reported systematically fewer edges than the Python side.
      mag[i] = Math.abs(gx) + Math.abs(gy);
      let ang = (Math.atan2(gy, gx) * 180) / Math.PI;
      if (ang < 0) ang += 180;
      dir[i] = ang < 22.5 || ang >= 157.5 ? 0 : ang < 67.5 ? 1 : ang < 112.5 ? 2 : 3;
    }

  // 3. Non-maximum suppression along the gradient direction.
  const thin = new Float32Array(n);
  for (let y = 1; y < h - 1; y++)
    for (let x = 1; x < w - 1; x++) {
      const i = y * w + x;
      let a, b;
      switch (dir[i]) {
        case 0: a = mag[i - 1]; b = mag[i + 1]; break;
        case 1: a = mag[i - w + 1]; b = mag[i + w - 1]; break;
        case 2: a = mag[i - w]; b = mag[i + w]; break;
        default: a = mag[i - w - 1]; b = mag[i + w + 1];
      }
      thin[i] = mag[i] >= a && mag[i] >= b ? mag[i] : 0;
    }

  // 4 + 5. Double threshold, then hysteresis: weak edges survive only if they
  // connect to a strong one.
  const edge = new Uint8Array(n);
  const stack = [];
  for (let i = 0; i < n; i++) {
    if (thin[i] >= hi) { edge[i] = 2; stack.push(i); }
    else if (thin[i] >= lo) edge[i] = 1;
  }
  while (stack.length) {
    const i = stack.pop();
    const y = (i / w) | 0, x = i % w;
    for (let dy = -1; dy <= 1; dy++)
      for (let dx = -1; dx <= 1; dx++) {
        const yy = y + dy, xx = x + dx;
        if (yy < 0 || yy >= h || xx < 0 || xx >= w) continue;
        const j = yy * w + xx;
        if (edge[j] === 1) { edge[j] = 2; stack.push(j); }
      }
  }

  let count = 0;
  for (let i = 0; i < n; i++) if (edge[i] === 2) count++;
  return count / n;
}
