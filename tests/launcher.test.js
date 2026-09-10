const assert = require('node:assert/strict')
const test = require('node:test')
const vm = require('node:vm')
const launcher = require('../pinokio.js')

function menu(files = [], running = [], local = {}) {
  return launcher.menu(null, {
    exists: path => files.includes(path),
    running: path => running.includes(path),
    local: () => local
  })
}

test('an incomplete installation offers Install, not Start', async () => {
  assert.equal((await menu(['app/env']))[0].href, 'install.js')
})

test('a completed installation defaults to Start', async () => {
  const items = await menu(['app/env', 'app/.installed'])
  assert.equal(items[0].href, 'start.js')
  assert.equal(items[0].default, true)
})

test('reset remains visible after removing the environment', async () => {
  assert.equal((await menu([], ['reset.js']))[0].text, 'Resetting')
})

test('update takes precedence over its nested installation', async () => {
  assert.equal((await menu([], ['update.js', 'install.js']))[0].text, 'Updating')
})

test('running server opens its captured URL', async () => {
  const url = 'http://127.0.0.1:7861'
  assert.equal((await menu(['app/env', 'app/.installed'], ['start.js'], { url }))[0].href, url)
})

function activeSteps(script, platform, gpus, files = []) {
  const steps = []
  const context = { platform, gpus, exists: path => files.includes(path) }
  for (const step of require(`../${script}`).run) {
    if (step.when && !vm.runInNewContext(step.when.slice(2, -2), context)) continue
    steps.push(step)
    if (step.next === null) break
  }
  return steps
}

test('unsupported hardware never reaches an installer or server shell', () => {
  for (const script of ['install.js', 'start.js', 'torch.js']) {
    for (const [platform, gpus] of [['darwin', ['apple']], ['win32', ['amd']], ['linux', []]]) {
      const steps = activeSteps(script, platform, gpus)
      assert.equal(steps.length, 1)
      assert.equal(steps[0].method, 'notify')
    }
  }
})

test('hybrid NVIDIA systems select the CUDA installer', () => {
  for (const platform of ['win32', 'linux']) {
    const steps = activeSteps('torch.js', platform, ['amd', 'nvidia'])
    assert.equal(steps.length, 1)
    assert.match(steps[0].params.message[0], /whl\/cu128/)
    assert.doesNotMatch(steps[0].params.message[0], /--no-deps/)
  }
})

test('installation marker follows dependency and import validation', () => {
  const steps = activeSteps('install.js', 'win32', ['nvidia'])
  const torch = steps.findIndex(step => step.method === 'script.start')
  const verify = steps.findIndex(step => step.params.message?.includes('uv pip check'))
  const marker = steps.findIndex(step => step.method === 'fs.write')
  assert.ok(torch < verify && verify < marker)
})

test('start refuses a partially installed environment', () => {
  const steps = activeSteps('start.js', 'win32', ['nvidia'], ['app/env'])
  assert.deepEqual(steps.map(step => step.method), ['notify'])
})

test('startup URL capture returns the full local URL', () => {
  const steps = activeSteps('start.js', 'linux', ['nvidia'], ['app/env', 'app/.installed'])
  const pattern = steps[0].params.on[0].event
  const match = new RegExp(pattern.slice(1, -1)).exec('Running on local URL: http://127.0.0.1:7861')
  assert.equal(match[1], 'http://127.0.0.1:7861')
  assert.equal(steps[1].params.url, '{{input.event[1]}}')
})
