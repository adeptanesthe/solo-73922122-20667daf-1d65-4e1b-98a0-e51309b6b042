// 前端撤回逻辑功能测试：DOM 打桩 + 可控 fetch，验证
//  1) 求解期间修改输入 → 旧建议立即撤下（结果区清空）
//  2) 迟到的旧响应即使成功返回也绝不恢复显示
//  3) 正常求解 → 结果正常渲染
const fs = require('fs');
const path = require('path');
const html = fs.readFileSync(path.join(__dirname, '..', 'app', 'static', 'index.html'), 'utf8');
const pageJs = html.match(/<script>([\s\S]*)<\/script>/)[1];

// ---- DOM 打桩 ----
const store = {};
function el(id) {
  if (!store[id]) {
    const e = {
      id, checked: false, innerHTML: '',
      style: { display: 'none' }, dataset: {}, disabled: false,
      listeners: {},
      classList: { add() {}, remove() {} },
      addEventListener(ev, fn) { (this.listeners[ev] ??= []).push(fn); },
      appendChild() {},
    };
    let txt = '';
    Object.defineProperty(e, 'textContent', {   // 真实 DOM 赋值数字也会字符串化
      get: () => txt, set: x => { txt = String(x); },
    });
    let v = '0';
    Object.defineProperty(e, 'value', {   // 真实 DOM 中 input.value 恒为字符串
      get: () => v, set: x => { v = String(x); },
    });
    store[id] = e;
  }
  return store[id];
}
global.document = {
  getElementById: el,
  querySelector: () => el('tbody'),
  createElement: () => el('tr_' + Math.random()),
};
let fetchImpl = () => Promise.reject(new Error('fetch not stubbed'));
global.fetch = (...a) => fetchImpl(...a);

const META = { feature_names: ['A','B','C','D'], total_grams: 1000, proba_threshold: 0.7,
  n_trees: 5, max_depth: 3, tree_depths: [3,3,3,3,3], train_seed: 1 };
const SAMPLES = { feasible: { original: [250,250,350,150], lower: [0,0,0,0],
  upper: [1000,1000,1000,1000], locked: [false,true,false,false] },
  infeasible: { original: [150,130,620,100], lower: [0,0,620,0],
  upper: [150,1000,1000,1000], locked: [false,false,false,false] },
  tolerance_pass: { base: [700,100,100,100], tolerance: [30,30,30,30],
    locked: [false,true,false,false], base_probability: 0.97 },
  tolerance_fail: { base: [400,300,200,100], tolerance: [40,40,40,40],
    locked: [false,false,false,false], base_probability: 0.8 } };
const OK_RESULT = { status: 'ok', suggestion: [390,250,210,150], probability: 0.7969,
  changed_count: 2, total_change: 280, solve_ms: 50, optimality: '已证明最优…',
  per_ingredient: [
    { name:'A', original:250, new:390, delta:140, locked:false },
    { name:'B', original:250, new:250, delta:0, locked:true },
    { name:'C', original:350, new:210, delta:-140, locked:false },
    { name:'D', original:150, new:150, delta:0, locked:false } ] };
const TOL_PASS_RESULT = { status: 'pass', reason: '最差概率已证明不低于阈值 0.7。',
  worst: [707,100,123,70], min_probability: 0.8937058, base_probability: 0.9736,
  rational_min: [40248931014381061, 45035996273704960], region_size: 124806, solve_ms: 120,
  threshold: 0.7, optimality: '已证明…全部 124806 个配方均合格',
  per_ingredient: [
    { name:'A', base:700, worst:707, delta:7, tolerance:30, lower:670, upper:730, locked:false },
    { name:'B', base:100, worst:100, delta:0, tolerance:30, lower:100, upper:100, locked:true },
    { name:'C', base:100, worst:123, delta:23, tolerance:30, lower:70, upper:130, locked:false },
    { name:'D', base:100, worst:70, delta:-30, tolerance:30, lower:70, upper:130, locked:false } ] };
const TOL_FAIL_RESULT = { status: 'fail', reason: '容差域内存在合格概率低于 0.7 的配方（最低 0.302381）。',
  worst: [362,265,234,139], min_probability: 0.3023809, base_probability: 0.7969,
  rational_min: [1,3], region_size: 618461, solve_ms: 170, threshold: 0.7,
  optimality: '已证实不合格反例…',
  per_ingredient: TOL_PASS_RESULT.per_ingredient.map((p,i)=>({
    name:p.name, base:400-i*0, worst:[362,265,234,139][i], delta:[362,265,234,139][i]-[400,300,200,100][i],
    tolerance:40, lower:[360,260,160,60][i], upper:[440,340,240,140][i], locked:false })) };
const TOL_PENDING_RESULT = { status: 'pending', reason: '求解器未能判定（unknown，可能超时）…' };

// boot 在 eval 时同步发起 fetch，须提前打桩
fetchImpl = url => Promise.resolve({ json: async () => url.includes('meta') ? META : SAMPLES });

const driver = `
(async () => {
  const sleep = ms => new Promise(r => setTimeout(r, ms));
  const assert = (name, cond) => { console.log((cond ? '[PASS] ' : '[FAIL] ') + name); if (!cond) process.exitCode = 1; };

  await sleep(20);
  assert('启动后载入可行样例', document.getElementById('orig0').value == 250);

  // --- 场景 1：正常求解渲染 ---
  fetchImpl = () => Promise.resolve({ json: async () => OK_RESULT });
  await solve();
  assert('正常求解后 ok 面板显示', document.getElementById('panelOk').style.display === 'block');
  assert('概率已渲染', document.getElementById('okProb').textContent.includes('79.69'));

  // --- 场景 2：求解期间修改 → 立即撤下；迟到结果不恢复 ---
  let resolveStale;
  fetchImpl = () => new Promise(res => { resolveStale = () => res({ json: async () => OK_RESULT }); });
  const p = solve();                 // 发起求解（不等待）
  await sleep(5);
  onUserEdit();                      // 用户修改输入
  assert('修改后旧建议立即撤下', document.getElementById('panelOk').style.display === 'none');
  assert('状态提示已撤下', document.getElementById('status').textContent.includes('撤下'));
  resolveStale();                    // 迟到响应“绕过”abort 成功返回（竞态）
  await p;
  await sleep(5);
  assert('迟到结果未恢复 ok 面板', document.getElementById('panelOk').style.display === 'none');
  assert('迟到结果未改写状态', document.getElementById('status').textContent.includes('撤下'));

  // --- 场景 3：撤下后重新求解仍正常 ---
  fetchImpl = () => Promise.resolve({ json: async () => OK_RESULT });
  await solve();
  assert('重新求解后 ok 面板恢复', document.getElementById('panelOk').style.display === 'block');

  // --- 场景 4：无解分支渲染 ---
  fetchImpl = () => Promise.resolve({ json: async () => ({ status: 'infeasible', reason: '已证明无解', solve_ms: 30 }) });
  await solve();
  assert('无解面板显示', document.getElementById('panelInfeasible').style.display === 'block');
  assert('ok 面板保持隐藏', document.getElementById('panelOk').style.display === 'none');

  // ================= 容差核查 =================
  // --- 场景 5：启动后容差表单载入放行样例 ---
  assert('容差表单载入放行样例基准值', document.getElementById('tbase0').value == 700);
  assert('容差表单载入偏差值', document.getElementById('ttol0').value == 30);
  assert('容差锁定项已勾选', document.getElementById('tlock1').checked === true);
  assert('允许区间即时计算', document.getElementById('trange0').textContent === '[670, 730]');
  assert('锁定项区间显示锁定', document.getElementById('trange1').textContent.includes('锁定'));

  // --- 场景 6：放行结论渲染 ---
  fetchImpl = (url) => Promise.resolve({ json: async () => url.includes('tolerance') ? TOL_PASS_RESULT : OK_RESULT });
  await toleranceCheck();
  assert('放行面板显示', document.getElementById('panelTolPass').style.display === 'block');
  assert('最低概率已渲染', document.getElementById('tolPassMin').textContent.includes('89.37'));
  assert('可行域规模已渲染', document.getElementById('tolPassSize').textContent === '124806');
  assert('其余容差面板隐藏',
    document.getElementById('panelTolFail').style.display === 'none' &&
    document.getElementById('panelTolPending').style.display === 'none');

  // --- 场景 7：核查期间修改 → 立即撤下；迟到响应不恢复 ---
  let resolveTolStale;
  fetchImpl = () => new Promise(res => { resolveTolStale = () => res({ json: async () => TOL_PASS_RESULT }); });
  const tp = toleranceCheck();
  await sleep(5);
  onTolEdit();
  assert('修改后容差结论立即撤下', document.getElementById('panelTolPass').style.display === 'none');
  assert('容差状态提示已撤下', document.getElementById('tolStatus').textContent.includes('撤下'));
  resolveTolStale();                 // 迟到响应绕过 abort 返回
  await tp;
  await sleep(5);
  assert('迟到容差结果未恢复面板', document.getElementById('panelTolPass').style.display === 'none');
  assert('迟到容差结果未改写状态', document.getElementById('tolStatus').textContent.includes('撤下'));

  // --- 场景 8：撤下后重新核查 → 不合格反例渲染 ---
  fetchImpl = (url) => Promise.resolve({ json: async () => url.includes('tolerance') ? TOL_FAIL_RESULT : OK_RESULT });
  await toleranceCheck();
  assert('不合格反例面板显示', document.getElementById('panelTolFail').style.display === 'block');
  assert('反例概率已渲染', document.getElementById('tolFailMin').textContent.includes('30.24'));
  assert('放行面板保持隐藏', document.getElementById('panelTolPass').style.display === 'none');

  // --- 场景 9：待判分支渲染 ---
  fetchImpl = (url) => Promise.resolve({ json: async () => url.includes('tolerance') ? TOL_PENDING_RESULT : OK_RESULT });
  await toleranceCheck();
  assert('待判面板显示', document.getElementById('panelTolPending').style.display === 'block');
  assert('待判原因已渲染', document.getElementById('tolPendingReason').textContent.includes('未能判定'));

  // --- 场景 10：非法输入分支 ---
  fetchImpl = (url) => Promise.resolve({ json: async () => ({ status: 'invalid', reason: '基准配方总克数必须等于 1000' }) });
  await toleranceCheck();
  assert('容差非法输入面板显示', document.getElementById('panelTolInvalid').style.display === 'block');

  // --- 场景 11：两套 epoch 互不干扰；标签页切换 ---
  fetchImpl = (url) => Promise.resolve({ json: async () => url.includes('tolerance') ? TOL_PASS_RESULT : OK_RESULT });
  await toleranceCheck();
  assert('容差放行结论已恢复', document.getElementById('panelTolPass').style.display === 'block');
  onUserEdit();                     // 修改试配表单不应撤下容差结论
  assert('试配修改不影响容差结论', document.getElementById('panelTolPass').style.display === 'block');
  onTolEdit();                      // 修改容差表单不应影响试配面板状态
  await solve();
  assert('容差修改不影响试配求解', document.getElementById('panelOk').style.display === 'block');
  assert('试配 ok 面板显示时容差面板仍隐藏', document.getElementById('panelTolPass').style.display === 'none');
})();
`;

eval(pageJs + driver);
