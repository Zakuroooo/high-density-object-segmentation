import {
  INPUT, PROTO, letterbox, toTensor, decode, buildMask, cannyEdgeDensity,
} from "./yolo.js";

/* Thresholds mirror src/config.py. Changing one here without changing it there
 * would make the demo show a method the paper did not evaluate. */
const CONF = 0.25, IOU_NMS = 0.45;
const DENSE_CONF = 0.15, DENSE_IOU_NMS = 0.35;
const EDGE_DENSITY_THRESHOLD = 0.08, DENSE_COUNT_THRESHOLD = 10;

const PALETTE = [
  [42, 120, 214], [235, 104, 52], [27, 175, 122], [237, 161, 0],
  [232, 123, 164], [0, 131, 0], [74, 58, 167], [227, 73, 72],
];

const $ = (id) => document.getElementById(id);
let session = null;

function status(msg, busy = false) {
  const el = $("status");
  el.textContent = msg;
  el.className = busy ? "status busy" : "status";
}

async function loadModel() {
  status("Loading the model (45 MB) — first visit only, then it is cached…", true);
  ort.env.wasm.wasmPaths =
    "https://cdn.jsdelivr.net/npm/onnxruntime-web@1.20.1/dist/";
  ort.env.wasm.numThreads = 1;   // cross-origin isolation is not guaranteed here
  const t = performance.now();
  session = await ort.InferenceSession.create("yolov8s-seg-dense.onnx", {
    executionProviders: ["wasm"],
    graphOptimizationLevel: "all",
  });
  status(`Model ready (${((performance.now() - t) / 1000).toFixed(1)}s). ` +
         `Drop in a crowded photo.`);
  $("run").disabled = false;
}

async function infer(tensorData, conf, iouThr, lb, outW, outH) {
  const feeds = {
    images: new ort.Tensor("float32", tensorData, [1, 3, INPUT, INPUT]),
  };
  const out = await session.run(feeds);
  const names = Object.keys(out);
  // output0 is the detection tensor, output1 the prototypes — identify by rank
  // rather than by name, since exporters disagree about naming.
  const det = out[names.find((n) => out[n].dims.length === 3)];
  const proto = out[names.find((n) => out[n].dims.length === 4)];

  const dets = decode(det.data, conf, iouThr);
  const masks = dets.map((d) => buildMask(d, proto.data, lb, outW, outH));
  return { dets, masks };
}

function draw(canvas, img, masks) {
  canvas.width = img.naturalWidth;
  canvas.height = img.naturalHeight;
  const ctx = canvas.getContext("2d");
  ctx.drawImage(img, 0, 0);
  if (!masks.length) return;

  const im = ctx.getImageData(0, 0, canvas.width, canvas.height);
  const d = im.data;
  masks.forEach((m, idx) => {
    const [r, g, b] = PALETTE[idx % PALETTE.length];
    for (let i = 0; i < m.length; i++) {
      if (!m[i]) continue;
      const p = i * 4;
      d[p] = 0.55 * r + 0.45 * d[p];
      d[p + 1] = 0.55 * g + 0.45 * d[p + 1];
      d[p + 2] = 0.55 * b + 0.45 * d[p + 2];
    }
  });
  ctx.putImageData(im, 0, 0);
}

async function run() {
  const img = $("preview");
  if (!img.src || !img.naturalWidth) return;
  $("run").disabled = true;

  const W = img.naturalWidth, H = img.naturalHeight;
  const lb = letterbox(img);
  const tensor = toTensor(lb.ctx);

  status("Running the standard pass…", true);
  const t0 = performance.now();
  const base = await infer(tensor, CONF, IOU_NMS, lb, W, H);
  const tBase = performance.now() - t0;

  // Edge density is measured on the original image, not the letterboxed one —
  // grey padding has no edges and would dilute the ratio.
  const full = document.createElement("canvas");
  full.width = W; full.height = H;
  full.getContext("2d", { willReadFrequently: true }).drawImage(img, 0, 0);
  const ed = cannyEdgeDensity(
    full.getContext("2d", { willReadFrequently: true }), W, H);

  const dense = ed > EDGE_DENSITY_THRESHOLD ||
                base.dets.length >= DENSE_COUNT_THRESHOLD;

  let hybrid = base, tHybrid = tBase;
  if (dense) {
    status("Scene looks crowded — running the relaxed pass…", true);
    const t1 = performance.now();
    const relaxed = await infer(tensor, DENSE_CONF, DENSE_IOU_NMS, lb, W, H);
    tHybrid = tBase + (performance.now() - t1);
    if (relaxed.dets.length > base.dets.length) hybrid = relaxed;
  }

  draw($("canvasBase"), img, base.masks);
  draw($("canvasHybrid"), img, hybrid.masks);

  $("nBase").textContent = base.dets.length;
  $("nHybrid").textContent = hybrid.dets.length;
  $("tBase").textContent = `${tBase.toFixed(0)} ms`;
  $("tHybrid").textContent = `${tHybrid.toFixed(0)} ms`;

  $("gate").innerHTML = dense
    ? `Edge density <code>${ed.toFixed(4)}</code> &gt; <code>0.08</code> — ` +
      `<strong>gate fired</strong>, the relaxed pass ran and recovered ` +
      `<strong>${hybrid.dets.length - base.dets.length}</strong> additional instance(s).`
    : `Edge density <code>${ed.toFixed(4)}</code> ≤ <code>0.08</code> and only ` +
      `${base.dets.length} detections — <strong>gate did not fire</strong>. ` +
      `The hybrid is identical to the detector here, which is the intended behaviour.`;

  $("results").hidden = false;
  status("Done.");
  $("run").disabled = false;
}

function useFile(file) {
  const img = $("preview");
  img.onload = () => { $("run").disabled = !session; run(); };
  img.src = URL.createObjectURL(file);
  $("dropHint").hidden = true;
  img.hidden = false;
}

function wire() {
  const drop = $("drop");
  $("file").addEventListener("change", (e) => {
    if (e.target.files[0]) useFile(e.target.files[0]);
  });
  drop.addEventListener("dragover", (e) => {
    e.preventDefault(); drop.classList.add("over");
  });
  drop.addEventListener("dragleave", () => drop.classList.remove("over"));
  drop.addEventListener("drop", (e) => {
    e.preventDefault(); drop.classList.remove("over");
    if (e.dataTransfer.files[0]) useFile(e.dataTransfer.files[0]);
  });
  $("run").addEventListener("click", run);

  document.querySelectorAll(".sample").forEach((el) => {
    el.addEventListener("click", async () => {
      const img = $("preview");
      img.onload = () => { $("run").disabled = !session; run(); };
      img.crossOrigin = "anonymous";
      img.src = el.dataset.src;
      $("dropHint").hidden = true;
      img.hidden = false;
    });
  });

  loadModel().catch((e) => status(`Model failed to load: ${e.message}`));
}

wire();
