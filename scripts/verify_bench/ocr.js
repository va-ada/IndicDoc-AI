// Read every benchmark page with Tesseract (tesseract.js = Tesseract 5 LSTM compiled to WebAssembly; the product
// calls the Tesseract binary with the same tessdata_best models). Writes words with boxes and block / paragraph /
// line numbers, the same fields as ml.ocr.read_words().
// Usage: node scripts/verify_bench/ocr.js data/verify_bench [workers]
const fs = require('fs');
const path = require('path');
const { createWorker } = require('tesseract.js');

// Verification mode of ml.ocr.extract_document(number_pass=True): Indic model alone + an English number pass
const LANGS = { src: 'eng', hi: 'hin', ta: 'tam' };

(async () => {
  const root = path.resolve(process.argv[2] || 'data/verify_bench');
  const nWorkers = parseInt(process.argv[3] || '4', 10);
  const langPath = path.join(__dirname, 'tessdata');
  const outFile = path.join(root, 'ocr.json');
  const done = fs.existsSync(outFile) ? JSON.parse(fs.readFileSync(outFile)) : {};
  // Translated pages are also read by the English model: its digits fix the Indic reading (number pass)
  const all = fs.readdirSync(path.join(root, 'pages')).filter((f) => f.endsWith('.png')).sort();
  const jobs = [];
  for (const f of all) {
    const lang = f.includes('_src_') ? LANGS.src : LANGS[f.slice(0, 2)];
    jobs.push([f, lang, f]);
    if (lang !== 'eng') jobs.push([f, 'eng', f + '#eng']);
  }
  const images = jobs.filter(([, , key]) => !(key in done));
  const groups = {};
  for (const job of images) (groups[job[1]] = groups[job[1]] || []).push(job);

  let count = 0;
  const started = Date.now();
  for (const [lang, files] of Object.entries(groups)) {
    const workers = await Promise.all([...Array(nWorkers)].map(() =>
      createWorker(lang, 1, { langPath, gzip: true, cachePath: path.join(root, '.tesscache') })));
    await Promise.all(workers.map((w) => w.setParameters({ tessedit_pageseg_mode: '3', preserve_interword_spaces: '1' })));
    let next = 0;
    await Promise.all(workers.map(async (worker) => {
      while (next < files.length) {
        const [f, , key] = files[next++];
        const { data } = await worker.recognize(path.join(root, 'pages', f), {}, { blocks: true });
        const words = [];
        (data.blocks || []).forEach((b, bi) => b.paragraphs.forEach((p, pi) => p.lines.forEach((l, li) =>
          l.words.forEach((w) => {
            if (!w.text.trim()) return;
            words.push({ text: w.text.trim(), conf: w.confidence, left: w.bbox.x0, top: w.bbox.y0,
              width: w.bbox.x1 - w.bbox.x0, height: w.bbox.y1 - w.bbox.y0,
              block_num: bi + 1, par_num: pi + 1, line_num: li + 1 });
          }))));
        done[key] = words;
        if (++count % 25 === 0) {
          fs.writeFileSync(outFile, JSON.stringify(done));
          console.log(`${count}/${images.length} pages, ${((Date.now() - started) / count / 1000).toFixed(2)} s/page`);
        }
      }
    }));
    await Promise.all(workers.map((w) => w.terminate()));
  }
  fs.writeFileSync(outFile, JSON.stringify(done));
  console.log(`OCR done: ${count} pages in ${((Date.now() - started) / 1000).toFixed(0)} s`);
})();
