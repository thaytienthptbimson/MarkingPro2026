'use strict';
/* OMR – Phiếu trả lời trắc nghiệm THPT 2026 | Client-side, OpenCV.js */

// ============ 1. HẰNG SỐ CẤU HÌNH (tinh chỉnh tại đây) ============
const OPENCV_URL = 'https://docs.opencv.org/4.x/opencv.js';
const SHEET  = { W: 1000, H: 1414 };   // kích thước phiếu sau khi nắn phẳng (px), tỉ lệ A4
const DETECT = { blur: 5, cannyLo: 50, cannyHi: 150, minAreaRatio: 0.25, epsilon: 0.02 };
const MARK   = { fillMin: 0.40, doubleRatio: 0.80 };  // ngưỡng tô / ngưỡng nghi tô 2 ô

/* Mỗi vùng là một lưới bong bóng. Tâm ô (hàng i, cột j) của lưới đặt tại origin:
     x = ox + j*dx ,  y = oy + i*dy   (đơn vị px trên ảnh đã nắn SHEET.W x SHEET.H)
   r = bán kính ROI (nên nhỏ hơn bán kính ô in một chút để không dính viền).
   !!! TỌA ĐỘ DƯỚI ĐÂY LÀ MINH HỌA – hãy đo lại trên phiếu thật rồi chỉnh theo vòng tròn debug. */
const DIGITS = '0123456789'.split('');
const SBD   = { rows: 10, cols: 8, dx: 34, dy: 28, r: 9, origins: [[70, 170]],  symbols: DIGITS };
const MADE  = { rows: 10, cols: 4, dx: 34, dy: 28, r: 9, origins: [[420, 170]], symbols: DIGITS };
const PART1 = { rows: 10, cols: 4, dx: 36, dy: 28, r: 9, symbols: ['A', 'B', 'C', 'D'],      // 4 khối x 10 câu = 40 câu
                origins: [[70, 420], [290, 420], [510, 420], [730, 420]] };
const PART2 = { rows: 4, cols: 2, dx: 36, dy: 24, r: 9, symbols: ['Đ', 'S'],                 // 8 câu, mỗi câu 4 ý a-d
                origins: [[70, 740], [290, 740], [510, 740], [730, 740], [70, 850], [290, 850], [510, 850], [730, 850]] };
const PART3 = { rows: 12, cols: 4, dx: 30, dy: 26, r: 9, symbols: ['-', ',', ...DIGITS],      // 6 câu, mỗi câu 4 cột
                origins: [[60, 990], [220, 990], [380, 990], [540, 990], [700, 990], [860, 990]] };

// Đáp án theo mã đề (tùy chọn). Ví dụ: '0101': { part1: ['A','C',...], part2: [['Đ','S','Đ','S'],...], part3: ['12','-3,5',...] }
const ANSWER_KEYS = {};
const SCORING = { part1: 0.25, part3: 0.5, part2: [0, 0.1, 0.25, 0.5, 1] };  // part2: theo số ý đúng 0..4

// ============ 2. TIỆN ÍCH OPENCV ============
const $ = id => document.getElementById(id);
const DEBUG = [];   // các ô đã đọc: {x, y, r, ratio} để vẽ overlay

function loadOpenCV() {
  return new Promise((resolve, reject) => {
    const s = document.createElement('script');
    s.src = OPENCV_URL; s.async = true;
    s.onerror = () => reject(new Error('Không tải được OpenCV.js (kiểm tra mạng).'));
    s.onload = () => {
      if (cv instanceof Promise) cv.then(m => { window.cv = m; resolve(); });  // bản build trả về Promise
      else if (cv.Mat) resolve();
      else cv.onRuntimeInitialized = resolve;
    };
    document.head.appendChild(s);
  });
}

// Sắp 4 góc theo thứ tự: trái-trên, phải-trên, phải-dưới, trái-dưới
function orderCorners(p) {
  const sum = q => q.x + q.y, diff = q => q.y - q.x;
  const tl = p.reduce((a, b) => sum(b) < sum(a) ? b : a), br = p.reduce((a, b) => sum(b) > sum(a) ? b : a);
  const tr = p.reduce((a, b) => diff(b) < diff(a) ? b : a), bl = p.reduce((a, b) => diff(b) > diff(a) ? b : a);
  let c = [tl, tr, br, bl];
  const d = (a, b) => Math.hypot(a.x - b.x, a.y - b.y);
  if (d(c[0], c[1]) > d(c[0], c[3])) c = [c[1], c[2], c[3], c[0]];  // phiếu chụp ngang -> xoay về dọc
  return c;
}

// ============ 3. PIPELINE XỬ LÝ ẢNH ============
// Bước 1-3: Grayscale -> Gaussian Blur -> Canny -> tìm contour 4 đỉnh lớn nhất (tờ phiếu)
function detectSheetCorners(src) {
  const gray = new cv.Mat(), blur = new cv.Mat(), edges = new cv.Mat(), dil = new cv.Mat();
  const kernel = cv.getStructuringElement(cv.MORPH_RECT, new cv.Size(3, 3));
  const contours = new cv.MatVector(), hier = new cv.Mat();
  let corners = null;
  try {
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, blur, new cv.Size(DETECT.blur, DETECT.blur), 0);
    cv.Canny(blur, edges, DETECT.cannyLo, DETECT.cannyHi);
    cv.dilate(edges, dil, kernel);                       // nối các nét viền bị đứt
    cv.findContours(dil, contours, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    const minArea = src.rows * src.cols * DETECT.minAreaRatio;
    let bestArea = 0;
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i), area = cv.contourArea(c);
      if (area > minArea && area > bestArea) {
        const ap = new cv.Mat();
        cv.approxPolyDP(c, ap, DETECT.epsilon * cv.arcLength(c, true), true);
        if (ap.rows === 4 && cv.isContourConvex(ap)) {
          corners = Array.from({ length: 4 }, (_, k) => ({ x: ap.data32S[2 * k], y: ap.data32S[2 * k + 1] }));
          bestArea = area;
        }
        ap.delete();
      }
      c.delete();
    }
  } finally {
    [gray, blur, edges, dil, kernel, contours, hier].forEach(m => m.delete());
  }
  return corners && orderCorners(corners);
}

// Bước 4: Perspective Transform
function warpSheet(src, c) {
  /* Ảnh chụp bị nghiêng/méo phối cảnh nên tờ phiếu là một tứ giác bất kỳ.
     - 'from': 4 góc tìm được trên ảnh gốc (TL, TR, BR, BL).
     - 'to'  : 4 góc của hình chữ nhật chuẩn SHEET.W x SHEET.H.
     getPerspectiveTransform giải ma trận homography 3x3 M ánh xạ from -> to;
     warpPerspective áp M lên từng pixel để "kéo phẳng" tờ phiếu, nhờ đó các tọa độ
     cố định trong hằng số (SBD, PART1...) luôn trùng đúng vị trí ô trên giấy. */
  const from = cv.matFromArray(4, 1, cv.CV_32FC2, c.flatMap(p => [p.x, p.y]));
  const to = cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, SHEET.W, 0, SHEET.W, SHEET.H, 0, SHEET.H]);
  const M = cv.getPerspectiveTransform(from, to);
  const dst = new cv.Mat();
  try {
    cv.warpPerspective(src, dst, M, new cv.Size(SHEET.W, SHEET.H), cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());
  } finally { from.delete(); to.delete(); M.delete(); }
  return dst;
}

// Bước 5: Threshold OTSU. THRESH_BINARY_INV: nét mực/ô tô -> trắng (255), nền giấy -> đen (0)
function binarize(warped) {
  const gray = new cv.Mat(), bin = new cv.Mat();
  try {
    cv.cvtColor(warped, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, gray, new cv.Size(5, 5), 0);
    cv.threshold(gray, bin, 0, 255, cv.THRESH_BINARY_INV | cv.THRESH_OTSU);  // OTSU tự chọn ngưỡng theo độ sáng ảnh
  } finally { gray.delete(); }
  return bin;
}

// Bước 6: Pixel Intensity
function fillRatio(bin, cx, cy, r) {
  /* Cắt ROI hình vuông cạnh 2r quanh tâm ô, đếm pixel trắng (= pixel mực) bằng countNonZero,
     chia cho diện tích ROI => tỉ lệ lấp đầy 0..1.
     Ô trống (chỉ có viền in nhỏ/bị cắt ngoài ROI) cho tỉ lệ rất thấp; ô tô kín cho tỉ lệ cao.
     Ngưỡng MARK.fillMin lọc nhiễu và vết tẩy mờ; nếu 2 ô cùng vượt ngưỡng (xem pickMark) thì coi là tô đúp. */
  const x = Math.round(cx - r), y = Math.round(cy - r), s = 2 * r;
  if (x < 0 || y < 0 || x + s > bin.cols || y + s > bin.rows) return 0;
  const roi = bin.roi(new cv.Rect(x, y, s, s));   // roi chỉ là view, vẫn phải delete
  const n = cv.countNonZero(roi);
  roi.delete();
  return n / (s * s);
}

function readBlock(bin, spec, ox, oy) {
  const m = [];
  for (let i = 0; i < spec.rows; i++) {
    m.push([]);
    for (let j = 0; j < spec.cols; j++) {
      const cx = ox + j * spec.dx, cy = oy + i * spec.dy, ratio = fillRatio(bin, cx, cy, spec.r);
      m[i].push(ratio);
      DEBUG.push({ x: cx, y: cy, r: spec.r, ratio });
    }
  }
  return m;
}

// Chọn ô được tô trong 1 nhóm: -1 = bỏ trống, -2 = tô nhiều ô, >=0 = chỉ số ô
function pickMark(vals) {
  const s = vals.map((v, i) => [v, i]).sort((a, b) => b[0] - a[0]);
  const v1 = s[0][0], v2 = s[1] ? s[1][0] : 0;
  if (v1 < MARK.fillMin) return -1;
  if (v2 >= MARK.fillMin && v2 >= v1 * MARK.doubleRatio) return -2;
  return s[0][1];
}
const sym = (spec, k) => k === -1 ? '' : k === -2 ? '*' : spec.symbols[k];
const readRows = (bin, spec, [ox, oy]) => readBlock(bin, spec, ox, oy).map(r => sym(spec, pickMark(r)));
const readCols = (bin, spec, [ox, oy]) => {
  const m = readBlock(bin, spec, ox, oy);
  return Array.from({ length: spec.cols }, (_, j) => sym(spec, pickMark(m.map(r => r[j])))).join('');
};

function readSheet(bin) {
  DEBUG.length = 0;
  return {
    sbd: readCols(bin, SBD, SBD.origins[0]),
    made: readCols(bin, MADE, MADE.origins[0]),
    part1: PART1.origins.flatMap(o => readRows(bin, PART1, o)),   // mỗi hàng = 1 câu, chọn A-D
    part2: PART2.origins.map(o => readRows(bin, PART2, o)),       // mỗi câu: 4 hàng a-d, chọn Đ/S
    part3: PART3.origins.map(o => readCols(bin, PART3, o)),       // mỗi câu: 4 cột, mỗi cột 1 ký tự
  };
}

function gradeSheet(r) {
  const key = ANSWER_KEYS[r.made];
  if (!key) return null;
  let score = 0;
  r.part1.forEach((a, i) => { if (a && a === (key.part1 || [])[i]) score += SCORING.part1; });
  r.part2.forEach((q, i) => { score += SCORING.part2[q.filter((a, k) => a && a === ((key.part2 || [])[i] || [])[k]).length]; });
  r.part3.forEach((a, i) => { if (a && a === (key.part3 || [])[i]) score += SCORING.part3; });
  return Math.round(score * 100) / 100;
}

// Vẽ phiếu đã nắn phẳng + vòng ROI (xanh = đã tô, đỏ = trống) để dễ canh tọa độ
function showDebug(warped) {
  const out = $('out');
  cv.imshow(out, warped);
  const ctx = out.getContext('2d');
  ctx.lineWidth = 2;
  DEBUG.forEach(b => {
    ctx.strokeStyle = b.ratio >= MARK.fillMin ? '#16a34a' : 'rgba(220,38,38,.6)';
    ctx.beginPath(); ctx.arc(b.x, b.y, b.r, 0, 2 * Math.PI); ctx.stroke();
  });
  out.hidden = false;
}

// Toàn bộ pipeline cho 1 ảnh trên canvas; mọi Mat đều được delete trong finally
function processCanvas(canvas) {
  const src = cv.imread(canvas);
  let warped = null, bin = null;
  try {
    const corners = detectSheetCorners(src);
    if (!corners) throw new Error('Không thấy đủ 4 góc phiếu. Đặt phiếu trên nền tối, lọt trọn trong khung.');
    warped = warpSheet(src, corners);
    bin = binarize(warped);
    const result = readSheet(bin);
    result.score = gradeSheet(result);
    showDebug(warped);
    return result;
  } finally {
    src.delete(); if (warped) warped.delete(); if (bin) bin.delete();
  }
}

// ============ 4. CAMERA & GIAO DIỆN ============
let stream = null;
function setStatus(msg, cls = '') { const el = $('status'); el.textContent = msg; el.className = cls; }

async function startCamera() {
  try {
    stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: { ideal: 'environment' }, width: { ideal: 1920 }, height: { ideal: 1080 } }, audio: false });
    $('video').srcObject = stream;
    await $('video').play();
    $('btnScan').disabled = false;
    setStatus('Camera sẵn sàng. Đưa phiếu vào khung rồi bấm "Chụp và chấm".', 'ok');
  } catch (e) {
    setStatus('Không mở được camera (cần HTTPS và quyền truy cập): ' + e.message, 'err');
  }
}

function renderResult(r) {
  const p2 = r.part2.map((q, i) => `  Câu ${i + 1}: ` + q.map((a, k) => 'abcd'[k] + ':' + (a || '_')).join(' ')).join('\n');
  const lines = [
    `SBD: ${r.sbd || '_'}`, `Mã đề: ${r.made || '_'}`,
    `Phần I:  ` + r.part1.map((a, i) => `${i + 1}${a || '_'}`).join(' '),
    `Phần II:\n${p2}`,
    `Phần III: ` + r.part3.map((a, i) => `${i + 1}=[${a || '_'}]`).join(' '),
    r.score === null ? 'Điểm: chưa có đáp án cho mã đề này (khai báo trong ANSWER_KEYS)' : `Điểm: ${r.score}`,
    '(_ = bỏ trống, * = tô nhiều ô)'];
  const pre = $('result'); pre.textContent = lines.join('\n'); pre.hidden = false;
}

function scan(canvas) {
  try {
    renderResult(processCanvas(canvas));
    setStatus('Đã quét xong.', 'ok');
  } catch (e) { setStatus(e.message, 'err'); }
}

function captureFrame() {
  const v = $('video'), c = $('cap');
  c.width = v.videoWidth; c.height = v.videoHeight;
  c.getContext('2d').drawImage(v, 0, 0);
  scan(c);
}

function loadImageFile(file) {
  const img = new Image();
  img.onload = () => {
    const c = $('cap'); c.width = img.naturalWidth; c.height = img.naturalHeight;
    c.getContext('2d').drawImage(img, 0, 0);
    URL.revokeObjectURL(img.src); scan(c);
  };
  img.src = URL.createObjectURL(file);
}

$('btnStart').onclick = startCamera;
$('btnScan').onclick = captureFrame;
$('file').onchange = e => { if (e.target.files[0] && window.cv && cv.Mat) loadImageFile(e.target.files[0]); };
window.addEventListener('pagehide', () => stream && stream.getTracks().forEach(t => t.stop()));

loadOpenCV().then(() => setStatus('OpenCV.js đã sẵn sàng.', 'ok'))
            .catch(e => setStatus(e.message, 'err'));
