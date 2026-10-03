const $ = s => document.querySelector(s), W = 1000, H = 1414, LS = 'omr2026';
// Cấu trúc điểm: n1/n2/n3 = số câu mỗi phần; s1/s3 = điểm mỗi câu Phần I / III. Phần II: 1 ý=0.1, 2 ý=0.25, 3 ý=0.5, 4 ý=1
const PRE = { toan: { n1: 12, n2: 4, n3: 6, s1: .25, s3: .5 }, khac: { n1: 18, n2: 4, n3: 6, s1: .25, s3: .25 }, anh: { n1: 40, n2: 0, n3: 0, s1: .25, s3: 0 } };
const P2PT = [0, .1, .25, .5, 1], P3ROWS = ['-', ',', ...'0123456789'];

// Khung chứa lưới ô tô, toạ độ chuẩn hoá 0..1 trên phiếu đã nắn thẳng (x,y,w,h), r = số hàng, c = số cột
const blk = (x, y, w, h, r, c) => ({ x, y, w, h, r, c });
const defCfg = () => ({
  name: blk(.05, .05, .55, .05, 1, 1),
  sbd: blk(.06, .14, .32, .15, 10, 8),
  made: blk(.46, .14, .14, .15, 10, 4),
  p1: [0, 1, 2, 3].map(i => blk(.05 + i * .235, .35, .17, .2, 10, 4)),
  p2: [0, 1, 2, 3, 4, 5, 6, 7].map(i => blk(.05 + (i % 4) * .235, .59 + (i > 3) * .1, .17, .08, 4, 2)),
  p3: [0, 1, 2, 3, 4, 5].map(i => blk(.05 + i * .155, .8, .12, .17, 12, 4))
});

// ---------- Cài đặt (lưu trong trình duyệt) ----------
let S = Object.assign({ exam: '', subj: 'toan', thr: .4, keys: '', roster: '', ocr: false, cfg: JSON.stringify(defCfg(), null, 1) }, JSON.parse(localStorage[LS] || '{}'));
const save = () => localStorage[LS] = JSON.stringify(S);
['exam', 'subj', 'thr', 'keys', 'roster', 'ocr', 'cfg'].forEach(id => {
  const el = $('#' + id), chk = el.type == 'checkbox';
  chk ? el.checked = S[id] : el.value = S[id];
  el.oninput = () => { S[id] = chk ? el.checked : id == 'thr' ? +el.value : el.value; save(); if (id == 'keys') keyInfo(); };
});
const getCfg = () => { try { return JSON.parse(S.cfg) } catch { return defCfg() } };
const esc = s => String(s).replace(/[&<>"]/g, c => '&#' + c.charCodeAt(0) + ';');
const st = (t, bad) => { $('#st').textContent = t; $('#st').className = bad ? 'bad' : '' };
const norm = s => String(s || '').replace(/\./g, ',').replace(/\s/g, '');

// ---------- Đáp án & danh sách ----------
function parseKeys(t) {
  const K = {}; let cur;
  t.split('\n').forEach(l => {
    l = l.trim(); if (!l) return;
    if (l[0] == '#') { cur = K[l.slice(1).trim()] = []; return }
    cur && cur.push(l);
  });
  for (const k in K) {
    const [a = '', b = '', c = ''] = K[k];
    K[k] = {
      p1: a.toUpperCase().replace(/[^ABCD-]/g, '').split('').map(x => x == '-' ? '' : x),
      p2: b.toUpperCase().replace(/Đ/g, 'D').split(/\s+/).filter(Boolean).map(g => g.replace(/[^DS-]/g, '').split('').map(x => x == 'D' ? 'Đ' : x == 'S' ? 'S' : '')),
      p3: c.split(';').map(norm).filter(Boolean)
    };
  }
  return K;
}
const parseRoster = () => Object.fromEntries(S.roster.split('\n').map(l => l.trim().split(/[;\t,]/)).filter(a => a.length > 1).map(a => [a[0].trim(), a.slice(1).join(' ').trim()]));
function keyInfo() {
  const K = parseKeys(S.keys), P = PRE[S.subj];
  $('#keyinfo').innerHTML = Object.keys(K).map(k => { const o = K[k], bad = o.p1.length != P.n1 || o.p2.length != P.n2 || o.p3.length != P.n3; return `<small class="${bad ? 'bad' : ''}">Mã ${esc(k)}: ${o.p1.length} / ${o.p2.length} / ${o.p3.length} câu${bad ? ` (cần ${P.n1} / ${P.n2} / ${P.n3})` : ''}</small>` }).join('<br>');
}
keyInfo();

// ---------- Nắn phiếu & đọc ô tô (OpenCV) ----------
const ready = (async () => { while (!(window.cv && cv.Mat)) await new Promise(r => setTimeout(r, 200)); })();
const dist = (a, b) => Math.hypot(a[0] - b[0], a[1] - b[1]);

function warp(cn) {
  const T = [], t = m => (T.push(m), m);
  try {
    const src = t(cv.imread(cn)), g = t(new cv.Mat()), e = t(new cv.Mat()), k = t(cv.Mat.ones(5, 5, cv.CV_8U)), cs = t(new cv.MatVector()), h = t(new cv.Mat());
    cv.cvtColor(src, g, cv.COLOR_RGBA2GRAY); cv.GaussianBlur(g, g, new cv.Size(5, 5), 0); cv.Canny(g, e, 50, 150); cv.dilate(e, e, k);
    cv.findContours(e, cs, h, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    let best = null, ba = src.rows * src.cols * .2;
    for (let i = 0; i < cs.size(); i++) {
      const c = cs.get(i), a = cv.contourArea(c);
      if (a > ba) { const ap = new cv.Mat(); cv.approxPolyDP(c, ap, .02 * cv.arcLength(c, true), true); if (ap.rows == 4) { best && best.delete(); best = ap; ba = a } else ap.delete() }
      c.delete();
    }
    let p = [[0, 0], [src.cols, 0], [src.cols, src.rows], [0, src.rows]], found = !!best;
    if (best) {
      const q = [0, 1, 2, 3].map(i => [best.data32S[i * 2], best.data32S[i * 2 + 1]]); best.delete();
      const by = f => q.reduce((m, v) => f(v) < f(m) ? v : m);
      p = [by(v => v[0] + v[1]), by(v => v[1] - v[0]), by(v => -(v[0] + v[1])), by(v => v[0] - v[1])]; // trái-trên, phải-trên, phải-dưới, trái-dưới
      if (dist(p[0], p[1]) > dist(p[0], p[3])) p = [p[3], p[0], p[1], p[2]]; // phiếu chụp ngang -> xoay lại
    }
    const M = t(cv.getPerspectiveTransform(t(cv.matFromArray(4, 1, cv.CV_32FC2, p.flat())), t(cv.matFromArray(4, 1, cv.CV_32FC2, [0, 0, W, 0, W, H, 0, H]))));
    const dst = t(new cv.Mat()); cv.warpPerspective(src, dst, M, new cv.Size(W, H));
    const out = document.createElement('canvas'); cv.imshow(out, dst);
    return { base: out, found };
  } finally { T.forEach(m => m.delete()) }
}

const cell = (b, i, j) => { const cw = b.w * W / b.c, ch = b.h * H / b.r; return [b.x * W + (j + .5) * cw, b.y * H + (i + .5) * ch, .3 * Math.min(cw, ch)] };
function fillAt(bin, x, y, r) {
  r = Math.max(2, r * .8 | 0); const x0 = Math.max(0, x - r | 0), y0 = Math.max(0, y - r | 0);
  const rc = bin.roi(new cv.Rect(x0, y0, Math.max(1, Math.min(2 * r, W - x0)), Math.max(1, Math.min(2 * r, H - y0))));
  const v = cv.mean(rc)[0] / 255; rc.delete(); return v;
}
const grid = (bin, b) => Array.from({ length: b.r }, (_, i) => Array.from({ length: b.c }, (_, j) => fillAt(bin, ...cell(b, i, j))));
// -1: bỏ trống, -2: tô nhiều ô, >=0: chỉ số ô được chọn
const pick = (v, thr) => { const m = Math.max(...v); if (m < thr) return -1; const i = v.indexOf(m); return v.some((x, k) => k != i && x >= thr && x > m * .75) ? -2 : i };

function readSheet(cn) {
  const T = [], t = m => (T.push(m), m);
  try {
    // Dùng kênh đỏ: mực in đỏ của phiếu biến mất, chỉ còn nét bút chì
    const s = t(cv.imread(cn)), pl = t(new cv.MatVector()); cv.split(s, pl);
    const R = t(pl.get(0)), bin = t(new cv.Mat());
    cv.GaussianBlur(R, R, new cv.Size(3, 3), 0);
    cv.adaptiveThreshold(R, bin, 255, cv.ADAPTIVE_THRESH_GAUSSIAN_C, cv.THRESH_BINARY_INV, 41, 15);
    const cfg = getCfg(), thr = S.thr, rd = { p1: [], p2: [], p3: [] };
    const cols = (b, f) => { const g = grid(bin, b); return Array.from({ length: b.c }, (_, j) => f(pick(g.map(r => r[j]), thr))) };
    const sbd = cols(cfg.sbd, k => k < 0 ? '?' : k).join(''), made = cols(cfg.made, k => k < 0 ? '?' : k).join('');
    cfg.p1.forEach(b => grid(bin, b).forEach(row => { const k = pick(row, thr); rd.p1.push(k >= 0 ? 'ABCD'[k] : k == -2 ? '?' : '') }));
    cfg.p2.forEach(b => rd.p2.push(grid(bin, b).map(([d, x]) => d >= thr && d > x * 1.3 ? 'Đ' : x >= thr && x > d * 1.3 ? 'S' : '')));
    cfg.p3.forEach(b => rd.p3.push(cols(b, k => k >= 0 ? P3ROWS[k] : '').join('')));
    return { rd, sbd, made };
  } finally { T.forEach(m => m.delete()) }
}

// ---------- Chấm điểm ----------
const recs = [];
function regrade(r) {
  const P = PRE[S.subj], K = parseKeys(S.keys)[r.made]; r.K = K;
  let a = 0, b = 0, c = 0;
  if (K) {
    for (let i = 0; i < P.n1; i++) a += K.p1[i] && r.rd.p1[i] === K.p1[i] ? P.s1 : 0;
    for (let i = 0; i < P.n2; i++) { let k = 0; for (let j = 0; j < 4; j++) k += K.p2[i] && K.p2[i][j] && r.rd.p2[i] && r.rd.p2[i][j] === K.p2[i][j] ? 1 : 0; b += P2PT[k] }
    for (let i = 0; i < P.n3; i++) c += K.p3[i] && K.p3[i] != '-' && norm(r.rd.p3[i]) === K.p3[i] ? P.s3 : 0;
  }
  r.parts = [a, b, c]; r.score = K ? Math.round((a + b + c) * 100) / 100 : '—';
  draw(r);
}

// Vẽ kết quả lên ảnh: xanh = đúng, đỏ = sai, vòng xanh dương = đáp án đúng
function draw(r) {
  const c = r.cv || (r.cv = document.createElement('canvas')); c.width = W; c.height = H;
  const x = c.getContext('2d'), cfg = getCfg(), K = r.K, P = PRE[S.subj];
  x.drawImage(r.base, 0, 0);
  const ring = (b, i, j, col, f) => { if (!b || j < 0) return; const [px, py, rd] = cell(b, i, j); x.beginPath(); x.arc(px, py, rd * 1.15, 0, 7); x.lineWidth = 3; x.strokeStyle = col; x.stroke(); if (f) { x.fillStyle = col + '55'; x.fill() } };
  const idx = (s, ch) => ch ? s.indexOf(ch) : -1;
  const mark = (b, i, s, k) => { if (s == k && s >= 0) ring(b, i, s, '#1f7a45', 1); else { ring(b, i, s, '#dd3333', 1); ring(b, i, k, '#1d5fbf') } };
  if (K) {
    for (let i = 0; i < P.n1; i++) mark(cfg.p1[i / 10 | 0], i % 10, idx('ABCD', r.rd.p1[i]), idx('ABCD', K.p1[i]));
    for (let i = 0; i < P.n2; i++) for (let j = 0; j < 4; j++) mark(cfg.p2[i], j, idx('ĐS', r.rd.p2[i] && r.rd.p2[i][j]), idx('ĐS', K.p2[i] && K.p2[i][j]));
    x.font = 'bold 22px sans-serif';
    for (let i = 0; i < P.n3; i++) { const b = cfg.p3[i]; if (!b) continue; const ok = norm(r.rd.p3[i]) === K.p3[i]; x.fillStyle = ok ? '#1f7a45' : '#dd3333'; x.fillText(ok ? '✓' : '✗ ' + K.p3[i], b.x * W, b.y * H - 6) }
  }
  x.fillStyle = 'rgba(255,255,255,.92)'; x.fillRect(0, 0, W, 64);
  x.fillStyle = '#23272e'; x.font = 'bold 26px sans-serif'; x.fillText(`${r.name || '(chưa có tên)'} · SBD ${r.sbd} · Mã ${r.made}`, 12, 40);
  x.fillStyle = '#c8372d'; x.font = 'bold 40px sans-serif'; x.textAlign = 'right'; x.fillText(K ? r.score + ' điểm' : 'Chưa có đáp án mã ' + r.made, W - 12, 46); x.textAlign = 'left';
  r.url = c.toDataURL('image/jpeg', .5);
}

async function ocrName(r) {
  const n = getCfg().name, c = document.createElement('canvas'); c.width = n.w * W; c.height = n.h * H;
  c.getContext('2d').drawImage(r.base, n.x * W, n.y * H, c.width, c.height, 0, 0, c.width, c.height);
  const { data } = await Tesseract.recognize(c, 'vie'); return data.text.replace(/\s+/g, ' ').trim();
}

async function addFile(f) {
  st('Đang chấm ' + f.name + '…'); await ready;
  const im = await createImageBitmap(f), sc = Math.min(1, 1800 / Math.max(im.width, im.height));
  const c = document.createElement('canvas'); c.width = im.width * sc; c.height = im.height * sc; c.getContext('2d').drawImage(im, 0, 0, c.width, c.height);
  const w = warp(c), r = { base: w.base, ...readSheet(w.base), name: '' };
  r.name = parseRoster()[r.sbd] || '';
  regrade(r); recs.push(r); render();
  if (!r.name && S.ocr && window.Tesseract) { st('Đang đọc chữ họ tên…'); try { r.name = await ocrName(r); regrade(r); render() } catch { } }
  st(w.found ? 'Xong. Kiểm tra lại tên và mã đề nếu có dấu “?”.' : 'Không thấy đủ 4 góc phiếu — hãy chụp rõ viền giấy trên nền tối.', !w.found);
}
['cam', 'gal'].forEach(id => $('#' + id).onchange = async e => {
  for (const f of e.target.files) { try { await addFile(f) } catch (err) { st('Lỗi: ' + err.message, 1) } }
  e.target.value = '';
});

// ---------- Danh sách, lưu ảnh, Excel ----------
function render() {
  $('#res').innerHTML = '<table><tr><th>STT</th><th>Ảnh</th><th>Họ tên</th><th>Mã đề</th><th>Điểm</th></tr>' +
    recs.map((r, i) => `<tr><td>${i + 1}</td><td><img class="th" data-i="${i}" src="${r.url}"></td><td><input data-n="${i}" value="${esc(r.name)}"><small>SBD ${esc(r.sbd)}</small></td><td><input data-m="${i}" value="${esc(r.made)}" size="4"></td><td><b>${r.score}</b></td></tr>`).join('') + '</table>';
}
$('#res').onclick = e => { const i = e.target.dataset.i; if (i != null) saveImg(+i) };
$('#res').onchange = e => {
  const d = e.target.dataset, r = recs[d.n ?? d.m]; if (!r) return;
  if (d.n != null) r.name = e.target.value.trim(); else r.made = e.target.value.trim();
  regrade(r); render();
};
function saveImg(i) {
  const r = recs[i];
  r.cv.toBlob(b => {
    const a = document.createElement('a'); a.href = URL.createObjectURL(b);
    a.download = `${i + 1}_${(r.name || r.sbd).replace(/[^\p{L}\p{N}]+/gu, '_')}_${r.score}.jpg`; a.click();
    setTimeout(() => URL.revokeObjectURL(a.href), 4000);
  }, 'image/jpeg', .9);
}
$('#allimg').onclick = async () => { for (let i = 0; i < recs.length; i++) { saveImg(i); await new Promise(r => setTimeout(r, 600)) } };
$('#xls').onclick = () => {
  if (!recs.length) return st('Chưa có bài nào để xuất.', 1);
  const wb = XLSX.utils.book_new();
  XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet([[S.exam || 'Kết quả thi'], ['STT', 'Tên học sinh', 'Điểm thi'], ...recs.map((r, i) => [i + 1, r.name, r.score])]), 'Điểm');
  XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet([['STT', 'SBD', 'Tên học sinh', 'Mã đề', 'Phần I', 'Phần II', 'Phần III', 'Tổng'], ...recs.map((r, i) => [i + 1, r.sbd, r.name, r.made, ...r.parts, r.score])]), 'Chi tiết');
  XLSX.writeFile(wb, (S.exam || 'ket-qua').replace(/[\\/:*?"<>|]+/g, '_') + '.xlsx');
};

// ---------- Căn chỉnh ----------
$('#grid').onclick = () => {
  const r = recs[recs.length - 1]; if (!r) return st('Hãy chấm thử 1 phiếu trước.', 1);
  const c = $('#cal'), x = c.getContext('2d'), cfg = getCfg(); c.width = W; c.height = H; x.drawImage(r.base, 0, 0);
  [cfg.name, cfg.sbd, cfg.made, ...cfg.p1, ...cfg.p2, ...cfg.p3].forEach(b => {
    x.strokeStyle = '#c8372d'; x.lineWidth = 2; x.strokeRect(b.x * W, b.y * H, b.w * W, b.h * H); x.fillStyle = '#1d5fbf';
    for (let i = 0; i < b.r; i++) for (let j = 0; j < b.c; j++) { const [px, py] = cell(b, i, j); x.fillRect(px - 2, py - 2, 4, 4) }
  });
};
$('#reread').onclick = async () => { await ready; recs.forEach(r => { Object.assign(r, readSheet(r.base)); r.name = parseRoster()[r.sbd] || r.name; regrade(r) }); render(); st('Đã đọc lại ' + recs.length + ' phiếu.') };
$('#reset').onclick = () => { S.cfg = $('#cfg').value = JSON.stringify(defCfg(), null, 1); save() };

// ---------- Xuất / nhập Excel cho đáp án (mục 2) và danh sách học sinh (mục 3) ----------
const pad = (v, n) => { const s = String(v).trim(); return /^\d+$/.test(s) && s.length < n ? s.padStart(n, '0') : s };
const readXls = async f => { const wb = XLSX.read(await f.arrayBuffer()); return XLSX.utils.sheet_to_json(wb.Sheets[wb.SheetNames[0]], { header: 1, defval: '' }) };
const setField = (id, v) => { S[id] = $('#' + id).value = v; save(); keyInfo(); renderKeys(); loadRoster() };
const dl = (rows, sheet, file, note) => {
  const wb = XLSX.utils.book_new(); XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(rows), sheet);
  XLSX.utils.book_append_sheet(wb, XLSX.utils.aoa_to_sheet(note.map(x => [x])), 'Hướng dẫn'); XLSX.writeFile(wb, file);
};

// Đáp án: cột A = nhãn câu (I.1, II.1a, III.1), các cột sau = mỗi mã đề một cột
$('#keyxls').onclick = () => {
  const P = PRE[S.subj], K = parseKeys(S.keys), ms = Object.keys(K).length ? Object.keys(K) : ['0101', '0102'];
  const rows = [['Câu', ...ms]], v = (m, f) => (K[m] ? f(K[m]) : '') || '';
  for (let i = 0; i < P.n1; i++) rows.push(['I.' + (i + 1), ...ms.map(m => v(m, k => k.p1[i]))]);
  for (let i = 0; i < P.n2; i++) for (let j = 0; j < 4; j++) rows.push(['II.' + (i + 1) + 'abcd'[j], ...ms.map(m => v(m, k => k.p2[i] && k.p2[i][j]))]);
  for (let i = 0; i < P.n3; i++) rows.push(['III.' + (i + 1), ...ms.map(m => v(m, k => k.p3[i]))]);
  dl(rows, 'Đáp án', 'dap-an-mau.xlsx', ['Mỗi cột là một mã đề (ô tiêu đề để dạng văn bản, ví dụ 0247).', 'Phần I: A, B, C hoặc D. Phần II: Đ hoặc S. Phần III: số, dùng dấu phẩy cho số thập phân (1,5).', 'Số câu theo môn đang chọn ở mục 1.']);
};
$('#keyfile').onchange = async e => {
  const f = e.target.files[0]; e.target.value = ''; if (!f) return;
  try {
    const a = await readXls(f), P = PRE[S.subj], n = getCfg().made.c, hd = a[0].slice(1).map(h => pad(h, n)), K = hd.map(() => ({ p1: [], p2: [], p3: [] }));
    a.slice(1).forEach(r => {
      const m = String(r[0]).toUpperCase().replace(/\s/g, '').match(/^(I{1,3})\.(\d+)([A-D])?$/); if (!m) return;
      const q = m[2] - 1;
      hd.forEach((_, k) => {
        const raw = String(r[k + 1]).trim(), u = raw.toUpperCase();
        if (m[1] == 'I') K[k].p1[q] = /^[ABCD]$/.test(u) ? u : '-';
        else if (m[1] == 'II') { if (m[3]) (K[k].p2[q] = K[k].p2[q] || [])['ABCD'.indexOf(m[3])] = u == 'S' ? 'S' : u == 'Đ' || u == 'D' ? 'Đ' : '-' }
        else K[k].p3[q] = norm(raw);
      });
    });
    const txt = hd.map((h, k) => h && ['#' + h,
      Array.from({ length: P.n1 }, (_, i) => K[k].p1[i] || '-').join(''),
      Array.from({ length: P.n2 }, (_, i) => [0, 1, 2, 3].map(j => (K[k].p2[i] || [])[j] || '-').join('')).join(' '),
      Array.from({ length: P.n3 }, (_, i) => K[k].p3[i] || '-').join('; ')].join('\n')).filter(Boolean).join('\n\n');
    if (!txt) throw new Error('không thấy mã đề (hàng 1) hoặc nhãn câu (I.1, II.1a, III.1) ở cột A.');
    if (S.keys.trim() && !confirm('Thay toàn bộ đáp án hiện có bằng file Excel?')) return;
    setField('keys', txt); st('Đã nhập đáp án của ' + hd.filter(Boolean).length + ' mã đề.');
  } catch (err) { st('Lỗi nhập Excel: ' + err.message, 1) }
};

// Danh sách học sinh: cột SBD và Họ và tên
$('#rosxls').onclick = () => {
  const R = Object.entries(parseRoster());
  dl([['SBD', 'Họ và tên'], ...(R.length ? R : [['01000001', 'Nguyễn Văn A'], ['01000002', 'Trần Thị B']])], 'Học sinh', 'danh-sach-hoc-sinh.xlsx', ['Giữ cột SBD ở dạng văn bản để không mất số 0 đầu.', 'Xóa các dòng ví dụ trước khi nhập.']);
};
$('#rosfile').onchange = async e => {
  const f = e.target.files[0]; e.target.value = ''; if (!f) return;
  try {
    const a = await readXls(f), h = a[0].map(x => String(x).toLowerCase());
    let c1 = h.findIndex(x => /sbd|báo danh/.test(x)), c2 = h.findIndex(x => /tên/.test(x)); const hasH = c1 >= 0 || c2 >= 0;
    if (c1 < 0) c1 = 0; if (c2 < 0) c2 = c1 == 0 ? 1 : 0;
    const n = getCfg().sbd.c, L = a.slice(hasH ? 1 : 0).map(r => [pad(r[c1], n), String(r[c2] ?? '').trim()]).filter(x => x[0] && x[1]).map(x => x.join(';'));
    if (!L.length) throw new Error('không đọc được dòng nào (cần cột SBD và Họ và tên).');
    if (S.roster.trim() && !confirm('Thay toàn bộ danh sách hiện có bằng file Excel?')) return;
    setField('roster', L.join('\n')); st('Đã nhập ' + L.length + ' học sinh.');
  } catch (err) { st('Lỗi nhập Excel: ' + err.message, 1) }
};

// ---------- Nhập đáp án (mục 2) dạng ô ----------
// Dữ liệu gốc vẫn là chuỗi S.keys; các ô chỉ là giao diện sửa chuỗi đó
let KM = {}, cur = '';
const ser = () => {
  const P = PRE[S.subj];
  return Object.keys(KM).map(m => {
    const k = KM[m];
    return ['#' + m,
      Array.from({ length: P.n1 }, (_, i) => k.p1[i] || '-').join(''),
      Array.from({ length: P.n2 }, (_, i) => [0, 1, 2, 3].map(j => (k.p2[i] || [])[j] || '-').join('')).join(' '),
      Array.from({ length: P.n3 }, (_, i) => k.p3[i] || '-').join('; ')].join('\n');
  }).join('\n\n');
};
const commit = () => { S.keys = $('#keys').value = ser(); save(); keyInfo() };

function renderKeys() {
  KM = parseKeys(S.keys); if (!KM[cur]) cur = Object.keys(KM)[0] || '';
  const P = PRE[S.subj], k = KM[cur];
  $('#chips').innerHTML = Object.keys(KM).map(m => `<button class="chip${m == cur ? ' on' : ''}" data-m="${esc(m)}">${esc(m)}</button>`).join('') + '<button class="chip" id="addm">＋ Mã đề</button>';
  if (!k) { $('#kgrid').innerHTML = '<small>Chưa có mã đề. Bấm “＋ Mã đề” để thêm.</small>'; return }
  let h = `<div class="row"><input id="q1" placeholder="Dán nhanh Phần I: ABCD…"><button class="alt" id="delm">Xóa mã ${esc(cur)}</button></div>`;
  if (P.n1) h += `<h4>Phần I – ${P.n1} câu</h4><div class="g">` + Array.from({ length: P.n1 }, (_, i) => `<div class="q"><b>${i + 1}</b>` + [...'ABCD'].map(c => `<button data-t="1" data-i="${i}" data-v="${c}"${k.p1[i] == c ? ' class="on"' : ''}>${c}</button>`).join('') + '</div>').join('') + '</div>';
  if (P.n2) h += `<h4>Phần II – ${P.n2} câu (4 ý a, b, c, d)</h4>` + Array.from({ length: P.n2 }, (_, i) => `<div class="q2"><b>Câu ${i + 1}</b>` + [0, 1, 2, 3].map(j => `<span>${'abcd'[j]}` + ['Đ', 'S'].map(v => `<button data-t="2" data-i="${i}" data-j="${j}" data-v="${v}"${(k.p2[i] || [])[j] == v ? ` class="on ${v == 'Đ' ? 'd' : 's'}"` : ''}>${v}</button>`).join('') + '</span>').join('') + '</div>').join('');
  if (P.n3) h += `<h4>Phần III – ${P.n3} câu (số, dùng dấu phẩy: 1,5)</h4><div class="g">` + Array.from({ length: P.n3 }, (_, i) => `<label class="q3"><b>${i + 1}</b><input data-t="3" data-i="${i}" inputmode="text" value="${esc(k.p3[i] && k.p3[i] != '-' ? k.p3[i] : '')}"></label>`).join('') + '</div>';
  $('#kgrid').innerHTML = h;
}
$('#subj').addEventListener('input', renderKeys);
$('#chips').onclick = e => {
  const b = e.target.closest('button'); if (!b) return;
  if (b.id == 'addm') {
    const m = prompt('Nhập mã đề (ví dụ 0247):'); if (!m || !m.trim()) return;
    cur = pad(m.trim(), getCfg().made.c); KM[cur] = KM[cur] || { p1: [], p2: [], p3: [] }; commit();
  } else cur = b.dataset.m;
  renderKeys();
};
$('#kgrid').onclick = e => {
  const b = e.target.closest('button'); if (!b || !KM[cur]) return;
  if (b.id == 'delm') { if (confirm('Xóa đáp án mã ' + cur + '?')) { delete KM[cur]; commit(); renderKeys() } return }
  const d = b.dataset, k = KM[cur], i = +d.i;
  if (d.t == '1') k.p1[i] = k.p1[i] == d.v ? '' : d.v;
  else if (d.t == '2') { k.p2[i] = k.p2[i] || []; k.p2[i][+d.j] = k.p2[i][+d.j] == d.v ? '' : d.v }
  else return;
  commit(); renderKeys();
};
$('#kgrid').oninput = e => { if (e.target.dataset.t == '3') { KM[cur].p3[+e.target.dataset.i] = norm(e.target.value); commit() } };
$('#kgrid').onchange = e => {
  if (e.target.id != 'q1') return;
  KM[cur].p1 = [...e.target.value.toUpperCase().replace(/[^ABCD]/g, '')]; commit(); renderKeys();
};

// ---------- Nhập SBD + họ tên (mục 3) dạng ô ----------
let R = [];
function loadRoster() { R = S.roster.split('\n').map(l => l.split(/;/)).filter(a => a[0].trim()).map(a => [a[0].trim(), (a[1] || '').trim()]); if (!R.length) R = [['', '']]; drawRoster() }
const rcommit = () => { S.roster = $('#roster').value = R.filter(r => r[0].trim()).map(r => r[0].trim() + ';' + r[1].trim()).join('\n'); save() };
function drawRoster() {
  $('#rgrid').innerHTML = R.map((r, i) => `<div class="rr"><span>${i + 1}</span><input data-r="${i}" data-c="0" value="${esc(r[0])}" inputmode="numeric" placeholder="SBD"><input data-r="${i}" data-c="1" value="${esc(r[1])}" placeholder="Họ và tên"><button data-del="${i}">✕</button></div>`).join('');
}
$('#rgrid').oninput = e => { const d = e.target.dataset; if (d.r != null) { R[+d.r][+d.c] = e.target.value; rcommit() } };
$('#rgrid').onclick = e => { const b = e.target.closest('button'); if (!b) return; R.splice(+b.dataset.del, 1); if (!R.length) R = [['', '']]; rcommit(); drawRoster() };
$('#addr').onclick = () => { R.push(['', '']); drawRoster(); const x = $('#rgrid').querySelectorAll('input'); x[x.length - 2].focus() };
// Dán nhiều dòng từ Excel: điền xuống từ ô đang chọn
$('#rgrid').onpaste = e => {
  const t = (e.clipboardData || window.clipboardData).getData('text'); if (!/[\n\t]/.test(t) || e.target.dataset.r == null) return;
  e.preventDefault(); const i = +e.target.dataset.r, c = +e.target.dataset.c, n = getCfg().sbd.c;
  t.trim().split(/\r?\n/).forEach((l, k) => {
    const a = l.split(/[\t;]/), o = R[i + k] || ['', ''];
    R[i + k] = c == 1 && a.length == 1 ? [o[0], a[0].trim()] : [pad(a[0], n), (a[1] || '').trim()];
  });
  rcommit(); drawRoster();
};

renderKeys(); loadRoster();
