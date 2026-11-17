// ISL Translator: camera -> MediaPipe landmarks -> 32-frame clip -> ONNX model -> word + speech.
// The landmark vector MUST follow docs/spec.md exactly (same as scripts/extract_landmarks.py).
import { FilesetResolver, HandLandmarker, PoseLandmarker } from "@mediapipe/tasks-vision";

const CFG = window.ISL_CFG;
const T = 32, NF = 142;
const POSE_IDX = [0, 11, 12, 13, 14, 15, 16];   // nose, shoulders, elbows, wrists
const ASPECT_TRAIN = 1920 / 1080;               // training videos are 16:9
const THRESHOLD = 0.5;                          // at or above this: accept
const MIN_LIKELY = 0.4, MARGIN = 2;             // or accept 40%+ when it is at least 2x the runner-up
const START_FRAMES = 3, STOP_SECONDS = 1.0, PAD = 8, PRE = 20, MAX_SECONDS = 6, MIN_HAND_FRAMES = 8;
const ACTIVE_Y = 1.2;   // a hand counts as "signing" when its wrist is above this line (shoulder-widths below the shoulders)

const $ = (id) => document.getElementById(id);
const video = $("video"), canvas = $("overlay"), ctx = canvas.getContext("2d");
const chip = $("chip"), btnCam = $("btnCam"), btnRec = $("btnRec"), btnHukum = $("btnHukum");

let hand, pose, session, labels = [];
let running = false, state = "idle";       // idle | countdown | recording
let buf = [], pre = [], handRun = 0, lastHandT = 0, startT = 0, manual = false;
let lastVideoTime = -1, tsCounter = 0;
let hukumArmed = false, hukumRun = false, countTimer = null;   // the Hukum button: only this path can play the animation
let fileMode = false, lastClip = null, lastInfo = "";

function setChip(text, rec = false) { chip.textContent = text; chip.classList.toggle("rec", rec); }

async function init() {
  setChip("Loading models...");
  const fileset = await FilesetResolver.forVisionTasks(CFG.wasm);
  const common = { runningMode: "VIDEO" };
  hand = await HandLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: CFG.hand }, numHands: 2,
    minHandDetectionConfidence: 0.3, minHandPresenceConfidence: 0.3, minTrackingConfidence: 0.3, ...common });
  pose = await PoseLandmarker.createFromOptions(fileset, {
    baseOptions: { modelAssetPath: CFG.pose }, numPoses: 1, ...common });
  session = await ort.InferenceSession.create("model/sign_model.onnx");
  labels = await (await fetch("model/labels.json")).json();
  $("collectWord").innerHTML = [...labels].sort((x, y) => x.localeCompare(y)).map((l) => `<option>${l}</option>`).join("");
  setChip("Ready");
  btnCam.disabled = false; $("btnFile").disabled = false;
  $("vocab").innerHTML = labels.filter((l) => !/hukum/i.test(l)).sort((x, y) => x.localeCompare(y)).map((l) => `<span>${l}</span>`).join("");
  setChip("Camera is off");               // nothing is captured or translated until the person presses a button
}

// ---- landmark vector (spec v1) ----
function frameFeatures(hr, pr, aspect) {
  if (!pr.landmarks || !pr.landmarks.length) return null;
  const k = aspect / ASPECT_TRAIN;               // express x as if the frame had the training aspect
  const pl = pr.landmarks[0];
  const pxy = POSE_IDX.map((i) => [pl[i].x * k, pl[i].y]);
  const cx = (pxy[1][0] + pxy[2][0]) / 2, cy = (pxy[1][1] + pxy[2][1]) / 2;
  const scale = Math.hypot(pxy[1][0] - pxy[2][0], pxy[1][1] - pxy[2][1]);
  if (scale < 1e-4) return null;
  const f = new Float32Array(NF);
  pxy.forEach((p, i) => { f[126 + 2 * i] = (p[0] - cx) / scale; f[127 + 2 * i] = (p[1] - cy) / scale; });

  const handed = hr.handedness || hr.handednesses || [];
  let hands = (hr.landmarks || []).map((lm, i) => ({
    label: handed[i][0].categoryName,
    pts: lm.map((p) => [p.x * k, p.y, p.z]) }));
  if (hands.length === 2 && hands[0].label === hands[1].label) {
    hands.sort((a, b) => a.pts[0][0] - b.pts[0][0]);
    hands[0].label = "Left"; hands[1].label = "Right";
  }
  const slots = {};
  hands.forEach((h) => { slots[h.label === "Left" ? 0 : 1] = h.pts; });
  for (const s of [0, 1]) {
    if (!slots[s]) continue;
    slots[s].forEach((p, j) => {
      const o = s * 63 + j * 3;
      f[o] = (p[0] - cx) / scale; f[o + 1] = (p[1] - cy) / scale; f[o + 2] = p[2] / scale;
    });
    f[140 + s] = 1;
  }
  return f;
}
const hasHand = (f) => f[140] > 0 || f[141] > 0;
// signing = a hand is up (wrist above the waist line). Resting hands clasped low do not count.
// The pose model always reports the wrists, even when the hand detector loses the hand (blur, hand in front of the face),
// so use them too. That way a sign is caught from the moment the arm starts to rise.
const wristUp = (f) => f[137] < ACTIVE_Y || f[139] < ACTIVE_Y;
const isActive = (f) => (f[140] > 0 && f[1] < ACTIVE_Y) || (f[141] > 0 && f[64] < ACTIVE_Y) || wristUp(f);

// ---- framing check: is the person positioned so the model can read them? ----
function framing(pr) {
  if (!pr.landmarks || !pr.landmarks.length) return { ok: false, msg: "Step into view" };
  const pl = pr.landmarks[0], nose = pl[0], ls = pl[11], rs = pl[12];
  const vis = (p) => (p.visibility === undefined ? 1 : p.visibility) > 0.5;
  if (!vis(nose) || !vis(ls) || !vis(rs)) return { ok: false, msg: "Show your head and both shoulders" };
  const w = Math.abs(ls.x - rs.x);
  const inside = (p) => p.x > 0.03 && p.x < 0.97 && p.y > 0.03 && p.y < 0.92;
  if (!inside(nose) || !inside(ls) || !inside(rs)) return { ok: false, msg: "Move so your shoulders are fully in view" };
  if (w < 0.12) return { ok: false, msg: "Move a little closer" };
  if (w > 0.6) return { ok: false, msg: "Step back a little" };
  return { ok: true, msg: "" };
}

// ---- inference ----
function resample(frames) {
  const out = new Float32Array(T * NF), n = frames.length;
  for (let t = 0; t < T; t++) {
    const i = Math.round((t * (n - 1)) / (T - 1));    // = np.linspace(0, n-1, 32).round()
    out.set(frames[i], t * NF);
  }
  return out;
}
function softmax(a) {
  const m = Math.max(...a), e = a.map((v) => Math.exp(v - m)), s = e.reduce((x, y) => x + y, 0);
  return e.map((v) => v / s);
}
async function predict(frames, show = true) {
  const clip = resample(frames);
  lastClip = clip; $("btnSave").disabled = false;
  if (show && $("collectOn").checked && !fileMode) addCollected(clip);
  const input = new ort.Tensor("float32", clip, [1, T, NF]);
  const logits = (await session.run({ clip: input })).logits.data;
  const p = softmax(Array.from(logits));
  const top = p.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  if (show) showResult(top);
  return top;
}
const isHukum = (i) => /hukum/i.test(labels[i]);
function showResult(top) {
  if (hukumRun) {                      // the Hukum button was used for this clip
    hukumRun = false;
    const h = top.find(([p, i]) => isHukum(i) && p >= 0.1);
    if (h) {
      $("word").textContent = labels[h[1]]; $("hint").textContent = "MVP sign accepted (" + (h[0] * 100).toFixed(0) + "%)";
      const sp = document.createElement("span"); sp.textContent = labels[h[1]]; $("history").appendChild(sp);
      hukumShow(); return;
    }
  }
  const [p1, i1] = top[0];
  const p2 = top[1] ? top[1][0] : 0;
  const strong = p1 >= THRESHOLD;
  const likely = !strong && p1 >= MIN_LIKELY && p1 >= MARGIN * p2;   // clear winner even if not very confident
  const sure = (strong || likely) && !isHukum(i1);   // Hukum never fires from live listening: use the Hukum button
  $("word").textContent = sure ? labels[i1] : "Not sure";
  $("hint").textContent = (isHukum(i1) ? "Use the MVP button for this one." : strong ? "Best match, " + (p1 * 100).toFixed(0) + "% confidence" : likely ? "Likely match, " + (p1 * 100).toFixed(0) + "% confidence" : "Low confidence - try again, slower and fully in view.") + (lastInfo ? "  [" + lastInfo + "]" : "");
  $("bars").innerHTML = top.map(([p, i]) =>
    `<div class="bar"><span class="name">${labels[i]}</span><span class="track"><span class="fill" style="width:${(p * 100).toFixed(0)}%"></span></span><span class="pct">${(p * 100).toFixed(0)}%</span></div>`).join("");
  if (sure && !$("collectOn").checked) {
    const h = document.createElement("span"); h.textContent = labels[i1]; $("history").appendChild(h);
    if (/hukum/i.test(labels[i1])) hukumShow();                      // easter egg: the glasses gesture
    else if ($("speak").checked && "speechSynthesis" in window) {
      speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance(labels[i1]));
    }
  }
}

// ---- easter egg: full-screen "Hukum!" animation (shown when the model recognises the "Hukum" gesture) ----
// Optional: put your own hukum.mp4 (or .webm, any length up to 60 seconds; the animation matches the clip) or hukum.jpg / .png / .webp in the app folder.
// It plays behind the text. Without those files the plain animation is shown.
const hukumMedia = { video: null, image: null };
(async () => {
  for (const f of ["hukum.mp4", "hukum.webm"]) {
    try { if ((await fetch(f, { method: "HEAD" })).ok) { hukumMedia.video = f; const pv = document.createElement("video"); pv.preload = "metadata"; pv.onloadedmetadata = () => { if (isFinite(pv.duration)) hukumMedia.dur = pv.duration; }; pv.src = f; break; } } catch (e) {}
  }
  if (!hukumMedia.video) for (const f of ["hukum.jpg", "hukum.png", "hukum.webp"]) {
    try { if ((await fetch(f, { method: "HEAD" })).ok) { hukumMedia.image = f; break; } } catch (e) {}
  }
})();
function hukumShow() {
  let o = document.getElementById("hukum");
  if (!o) {
    o = document.createElement("div"); o.id = "hukum";
    o.innerHTML = '<div class="bgwrap"></div><div class="h1">HUKUM!</div><div class="h2">Tiger ka Hukum</div>';
    o.addEventListener("click", () => { clearTimeout(hukumShow.t); o.classList.remove("go"); o.querySelector(".bgwrap").innerHTML = ""; });   // tap to close
    document.body.appendChild(o);
  }
  const bg = o.querySelector(".bgwrap");
  bg.innerHTML = "";
  let vid = null, dur = 3.4;
  if (hukumMedia.video) {
    vid = document.createElement("video"); vid.className = "bgmedia"; vid.src = hukumMedia.video; vid.playsInline = true;
    bg.appendChild(vid); dur = Math.min(60, (hukumMedia.dur || 11) + 0.3);
  } else if (hukumMedia.image) {
    const im = document.createElement("img"); im.className = "bgmedia"; im.src = hukumMedia.image; bg.appendChild(im); dur = 4;
  }
  o.style.setProperty("--hd", dur + "s");
  o.classList.remove("go"); void o.offsetWidth; o.classList.add("go");   // restart the animation
  if (vid) vid.play().catch(() => { vid.muted = true; vid.play().catch(() => {}); });
  clearTimeout(hukumShow.t);
  hukumShow.t = setTimeout(() => { o.classList.remove("go"); bg.innerHTML = ""; }, dur * 1000 + 100);
  if (!vid && $("speak").checked && "speechSynthesis" in window) {          // with a video the clip's own sound is used
    speechSynthesis.cancel(); speechSynthesis.speak(new SpeechSynthesisUtterance("Tiger ka hukum"));
  }
}

// ---- recording state machine ----
// "Start translating" turns on continuous detection (translating = true) until "Stop translating" is pressed.
// The MVP button is a one-shot manual capture: press, sign, press again.
let translating = false;
function startRecording(now) { state = "recording"; buf = pre.slice(); startT = now; lastHandT = now; setChip("Recording...", true); updateButtons(); }
function beginCapture() {                                         // MVP button: manual capture, no countdown
  if (!running || fileMode || translating || state !== "idle") return;
  hukumArmed = true; buf = []; pre = []; manual = true; startRecording(performance.now());
}
function toggleTranslating() {
  if (!running || fileMode) return;
  if (translating) {
    translating = false; handRun = 0; pre = [];
    if (state === "recording" && !manual) stopRecording("button");   // translate what was being signed, then stop
    setChip("Ready"); updateButtons();
  } else if (state === "idle") {
    translating = true; handRun = 0; pre = []; setChip("Translating - show a sign"); updateButtons();
  }
}
function updateButtons() {
  const ready = running && !fileMode;
  const idle = state === "idle", rec = state === "recording";
  btnRec.disabled = !ready || (hukumArmed && rec);
  btnHukum.disabled = !ready || translating || (!idle && !(rec && hukumArmed));
  btnRec.classList.toggle("live", translating);
  btnRec.querySelector(".t").textContent = translating ? "Stop translating" : "Start translating";
  btnHukum.querySelector(".t").textContent = rec && hukumArmed ? "Stop" : "MVP";
  $("btnStop").hidden = !ready; $("btnStop").disabled = translating || rec;
  $("stage").classList.toggle("on", ready);
  $("stage").classList.toggle("recording", rec);
}
async function stopRecording(reason) {
  hukumRun = hukumArmed; hukumArmed = false;
  state = "idle"; manual = false; handRun = 0; pre = [];
  updateButtons();
  // drop trailing frames without hands, keeping a small pad
  let last = buf.length - 1;
  while (last > 0 && !isActive(buf[last])) last--;
  const clip = buf.slice(0, Math.min(buf.length, last + 1 + PAD));
  buf = [];
  const handFrames = clip.filter(isActive).length;
  if (handFrames < MIN_HAND_FRAMES || clip.filter(hasHand).length < 4) {
    hukumRun = false; setChip(translating ? "Translating - show a sign" : running ? "Ready" : "Stopped");
    if (hukumRun === false && !translating) { $("word").textContent = "No sign seen"; $("hint").textContent = "Keep your head, shoulders and both hands in view, then try again."; $("bars").innerHTML = ""; }
    return;
  }
  setChip("Thinking...");
  lastInfo = `clip ${clip.length} frames, hand seen in ${clip.filter(hasHand).length}, hands up in ${handFrames}, both hands in ${clip.filter((x) => x[140] > 0 && x[141] > 0).length}`;
  console.log("CLIP", lastInfo);
  await predict(clip);
  setChip(translating ? "Translating - show a sign" : running ? "Ready" : "Stopped");
}
function onFrame(f, now, fr) {
  if (fileMode) { if (f) { buf.push(f); setChip("Reading video... " + buf.length + " frames", true); } return; }
  if (!f) { if (state === "idle") { handRun = 0; setChip(translating ? (fr && fr.msg ? fr.msg : "Step into view") : "Ready"); } return; }
  const h = isActive(f);
  if (state === "idle") {
    if (!translating) { setChip(fr.ok ? "Ready" : fr.msg); return; }       // not translating: nothing is captured
    pre.push(f); if (pre.length > PRE) pre.shift();
    handRun = h && fr.ok ? handRun + 1 : 0;
    if (handRun >= START_FRAMES) startRecording(now);
    else setChip(!fr.ok ? fr.msg : h ? "Hands up" : "Translating - show a sign");
  } else {
    buf.push(f);
    if (h) lastHandT = now;
    const quiet = (now - lastHandT) / 1000, dur = (now - startT) / 1000;
    if (!manual && quiet > STOP_SECONDS) stopRecording("quiet");
    else if (dur > MAX_SECONDS) stopRecording("max");
  }
}

// ---- drawing ----
function draw(hr, pr) {
  if (canvas.width !== video.videoWidth) { canvas.width = video.videoWidth; canvas.height = video.videoHeight; }
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  ctx.fillStyle = "#00e676";
  for (const lm of hr.landmarks || []) for (const p of lm) {
    ctx.beginPath(); ctx.arc(p.x * canvas.width, p.y * canvas.height, 3, 0, 7); ctx.fill();
  }
  ctx.fillStyle = "#ffca28";
  if (pr.landmarks && pr.landmarks[0]) for (const i of POSE_IDX) {
    const p = pr.landmarks[0][i];
    ctx.beginPath(); ctx.arc(p.x * canvas.width, p.y * canvas.height, 4, 0, 7); ctx.fill();
  }
}

function loop() {
  if (!running) return;
  if (video.readyState >= 2 && video.currentTime !== lastVideoTime) {
    lastVideoTime = video.currentTime;
    const ts = ++tsCounter * 33;                       // strictly increasing timestamps
    const hr = hand.detectForVideo(video, ts), pr = pose.detectForVideo(video, ts);
    draw(hr, pr);
    const fr = framing(pr);
    if (!fileMode) {
      $("guide").classList.toggle("ok", fr.ok); $("frameHint").textContent = fr.ok ? "Good position. Just sign, then pause." : fr.msg;
    }
    onFrame(frameFeatures(hr, pr, video.videoWidth / video.videoHeight), performance.now(), fr);
  }
  requestAnimationFrame(loop);
}

// ---- UI ----
async function startCamera() {
  setChip("Starting camera...");
  try {
    video.srcObject = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: "user", width: { ideal: 1280 }, height: { ideal: 720 }, frameRate: { ideal: 30 } }, audio: false });
  } catch (e) { setChip("Camera blocked"); $("camMsg").textContent = "Camera access was blocked. Allow the camera in your browser's address bar, then press Start camera again."; return; }
  await video.play();
  running = true; state = "idle"; translating = false; setChip("Ready");
  $("guide").classList.remove("off"); updateButtons();
  requestAnimationFrame(loop);
}
function stopCamera() {
  translating = false; hukumArmed = false; running = false; state = "idle"; buf = [];
  if (video.srcObject) video.srcObject.getTracks().forEach((t) => t.stop());
  video.srcObject = null; ctx.clearRect(0, 0, canvas.width, canvas.height);
  setChip("Camera is off"); $("guide").classList.add("off"); updateButtons();
}
btnCam.onclick = startCamera;
$("btnStop").onclick = stopCamera;
btnRec.onclick = toggleTranslating;
btnHukum.onclick = () => { if (state === "idle") beginCapture(); else if (state === "recording" && hukumArmed) stopRecording("button"); };
$("btnClear").onclick = () => { $("history").innerHTML = ""; };

window.__isl = { get state() { return state; } };   // handy for debugging in the console
init().catch((e) => { setChip("Failed to load"); console.error(e); });

// ---- test with a video file (same pipeline as the camera) ----
$("btnFile").onclick = () => $("fileIn").click();
$("fileIn").onchange = async (e) => {
  const file = e.target.files[0]; if (!file) return;
  if (running && video.srcObject) { video.srcObject.getTracks().forEach((t) => t.stop()); }
  video.srcObject = null; video.src = URL.createObjectURL(file); video.loop = false;
  fileMode = true; buf = []; state = "idle"; running = true; lastVideoTime = -1; $("guide").classList.add("off");
  updateButtons();
  await video.play();
  requestAnimationFrame(loop);
  video.onended = async () => {
    running = false; fileMode = false; lastInfo = ""; updateButtons();
    const full = buf.slice(); buf = [];
    setChip("Thinking...");
    const trimmed = (() => { let a = 0, b = full.length - 1;
      while (a < b && !hasHand(full[a])) a++; while (b > a && !hasHand(full[b])) b--;
      return full.slice(Math.max(0, a - PAD), Math.min(full.length, b + 1 + PAD)); })();
    const topT = await predict(trimmed, false);
    const top = await predict(full, false);          // last, so "Save last clip" saves the full clip
    // show the full-clip result (that is what training used) and log both
    showResult(top);
    console.log("FILE TEST", file.name, "frames", full.length,
      "full:", labels[top[0][1]], top[0][0].toFixed(2), "| trimmed:", labels[topT[0][1]], topT[0][0].toFixed(2));
    $("hint").textContent += `  (full clip: ${labels[top[0][1]]} ${(top[0][0]*100).toFixed(0)}%, trimmed: ${labels[topT[0][1]]} ${(topT[0][0]*100).toFixed(0)}%)`;
    setChip("Done");
    e.target.value = "";
  };
};

// ---- save the last clip (32 x 142) as JSON, for debugging and for collecting new training data ----
$("btnSave").onclick = () => {
  if (!lastClip) return;
  const label = $("clipLabel").value.trim() || "unlabelled";
  const blob = new Blob([JSON.stringify({ label, frames: T, features: NF, data: Array.from(lastClip) })],
                        { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `${label.replace(/[^a-z0-9]+/gi, "_")}_${Date.now()}.json`;
  a.click();
};

// Browsers only allow speech after the first tap on the page. Unlock it with a silent utterance.
const unlockVoice = () => {
  try { speechSynthesis.speak(new SpeechSynthesisUtterance("")); } catch (e) {}
  
  removeEventListener("pointerdown", unlockVoice); removeEventListener("keydown", unlockVoice);
};
addEventListener("pointerdown", unlockVoice); addEventListener("keydown", unlockVoice);

// ---- collect mode: record your own examples, then download them for retraining ----
const collected = [];
function renderCollect() {
  const counts = {};
  collected.forEach((c) => { counts[c.label] = (counts[c.label] || 0) + 1; });
  $("btnDl").textContent = `Download clips (${collected.length})`;
  $("btnDl").disabled = !collected.length; $("btnUndo").disabled = !collected.length;
  $("counts").textContent = Object.entries(counts).map(([k, v]) => `${k}: ${v}`).join("   |   ");
}
function addCollected(clip) {
  const label = $("collectNew").value.trim() || $("collectWord").value;   // typed new word wins over the list
  collected.push({ label, data: Array.from(clip, (v) => +v.toFixed(5)) });
  const n = collected.filter((c) => c.label === label).length;
  setChip(`Saved "${label}" #${n}`);
  $("collectInfo").textContent = `Saved "${label}" #${n}.  [${lastInfo}]  Wrong sign? Press Undo last.`;
  renderCollect();
}
$("btnUndo").onclick = () => { collected.pop(); renderCollect(); };
$("btnDl").onclick = () => {
  const blob = new Blob([JSON.stringify({ format: "isl-clips-v1", frames: T, features: NF, clips: collected })],
                        { type: "application/json" });
  const a = document.createElement("a");
  a.href = URL.createObjectURL(blob);
  a.download = `clips_${new Date().toISOString().slice(0, 16).replace(/[:T]/g, "-")}.json`;
  a.click();
};
addEventListener("beforeunload", (e) => { if (collected.length) { e.preventDefault(); e.returnValue = ""; } });
