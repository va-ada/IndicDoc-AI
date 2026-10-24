// Render benchmark papers to page images and record the box of every labelled element (the "oracle" layout).
// Usage: node scripts/verify_bench/render.js data/verify_bench
const fs = require('fs');
const path = require('path');
const { chromium } = require('playwright');

(async () => {
  const root = path.resolve(process.argv[2] || 'data/verify_bench');
  const htmlDir = path.join(root, 'html');
  const pngDir = path.join(root, 'pages');
  fs.mkdirSync(pngDir, { recursive: true });
  const browser = await chromium.launch();
  // 200 dpi, like the product renders PDF pages for verification (backend VERIFY_DPI): A4 = 1240 CSS px x 4/3
  const scale = 4 / 3;
  const page = await browser.newPage({ viewport: { width: 1240, height: 1754 }, deviceScaleFactor: scale });
  const layout = {};
  const files = fs.readdirSync(htmlDir).filter((f) => f.endsWith('.html')).sort();
  for (const file of files) {
    const name = file.replace('.html', '');
    await page.goto('file://' + path.join(htmlDir, file));
    await page.evaluate(() => document.fonts.ready);
    const pages = await page.$$('.page');
    layout[name] = [];
    for (let p = 0; p < pages.length; p++) {
      const el = pages[p];
      const info = await el.evaluate((node) => {
        const origin = node.getBoundingClientRect();
        const regions = [...node.querySelectorAll('[data-label]')].map((r) => {
          const b = r.getBoundingClientRect();
          return { label: r.dataset.label, box: [b.left - origin.left, b.top - origin.top, b.right - origin.left, b.bottom - origin.top] };
        });
        return { regions, overflow: node.scrollHeight > node.clientHeight + 1 };
      });
      info.regions.forEach((r) => { r.box = r.box.map((v) => v * scale); });
      const png = `${name}_p${p}.png`;
      await el.screenshot({ path: path.join(pngDir, png) });
      layout[name].push({ image: png, ...info });
      if (info.overflow) console.warn(`overflow: ${png}`);
    }
  }
  fs.writeFileSync(path.join(root, 'layout.json'), JSON.stringify(layout));
  await browser.close();
  console.log(`rendered ${files.length} papers`);
})();
