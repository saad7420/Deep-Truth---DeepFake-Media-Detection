<#
.SYNOPSIS
    Deep Truth — one-command setup and run (Windows 10/11, PowerShell 5.1+).

.DESCRIPTION
    Checks every dependency, installs what is missing where it safely can,
    then starts Redis-dependent services and reports health.

    Three things differ from the Linux script, and they are the reasons this
    file exists rather than a note in the README:

    1. Celery's default prefork pool CANNOT work on Windows — it needs fork(),
       which Windows does not have. The worker starts and then fails or hangs
       the moment it picks up a task. `--pool=solo` is mandatory here, and it
       runs one task at a time, so concurrency comes from running several
       workers rather than from --concurrency.

    2. There is no official Redis for Windows. This script detects Memurai,
       WSL, or Docker and guides accordingly.

    3. `import deeptruth_pipeline` needs the repo directory to carry that name.
       Symlinks on Windows need admin or Developer Mode, so the script prefers
       telling you to rename the folder over silently failing.

.EXAMPLE
    .\scripts\start.ps1              # check, install, run
    .\scripts\start.ps1 -Check       # report only
    .\scripts\start.ps1 -Stop        # stop everything it started
    .\scripts\start.ps1 -Restart     # required after changing Python code
    .\scripts\start.ps1 -Status
    .\scripts\start.ps1 -Workers 2   # two solo workers = two files at once
#>
[CmdletBinding()]
param(
    [switch]$Check,
    [switch]$Stop,
    [switch]$Restart,
    [switch]$Status,
    [switch]$Logs,
    [switch]$NoWeb,
    [switch]$Yes,
    [int]$Workers = 2
)

$ErrorActionPreference = 'Stop'

$RepoRoot  = Split-Path -Parent $PSScriptRoot
$ServerDir = Join-Path $RepoRoot 'server'
$ClientDir = Join-Path $RepoRoot 'client'
$RunDir    = Join-Path $RepoRoot '.run'
New-Item -ItemType Directory -Force -Path $RunDir | Out-Null

function Write-Step { param($m) Write-Host "`n$m" -ForegroundColor White }
function Write-Ok   { param($m) Write-Host "  [ok]   $m" -ForegroundColor Green }
function Write-Warn { param($m) Write-Host "  [warn] $m" -ForegroundColor Yellow }
function Write-Bad  { param($m) Write-Host "  [FAIL] $m" -ForegroundColor Red }
function Write-Info { param($m) Write-Host "  $m" -ForegroundColor DarkGray }

function Confirm-Action {
    param($Prompt)
    if ($Yes) { return $true }
    $r = Read-Host "  -> $Prompt [Y/n]"
    return ($r -eq '' -or $r -match '^[Yy]')
}

function Test-Port {
    param([int]$Port)
    try {
        $c = New-Object Net.Sockets.TcpClient
        $c.Connect('127.0.0.1', $Port); $c.Close(); return $true
    } catch { return $false }
}

# ─── process tracking ────────────────────────────────────────────────────────
# PID files, not name matching: killing "every python.exe" would take out
# whatever else the developer happens to be running.

function Get-TrackedPid { param($Name)
    $f = Join-Path $RunDir "$Name.pid"
    if (Test-Path $f) { return [int](Get-Content $f -First 1) }
    return $null
}

function Test-Tracked { param($Name)
    $procId = Get-TrackedPid $Name
    if (-not $procId) { return $false }
    return [bool](Get-Process -Id $procId -ErrorAction SilentlyContinue)
}

function Start-Tracked {
    param($Name, $WorkDir, $Exe, [string[]]$Arguments)
    if (Test-Tracked $Name) {
        Write-Ok "$Name already running (pid $(Get-TrackedPid $Name))"
        return
    }
    $log = Join-Path $RunDir "$Name.log"
    $p = Start-Process -FilePath $Exe -ArgumentList $Arguments `
                       -WorkingDirectory $WorkDir `
                       -RedirectStandardOutput $log `
                       -RedirectStandardError (Join-Path $RunDir "$Name.err.log") `
                       -WindowStyle Hidden -PassThru
    $p.Id | Set-Content (Join-Path $RunDir "$Name.pid")
    Start-Sleep -Seconds 1
    if (Get-Process -Id $p.Id -ErrorAction SilentlyContinue) {
        Write-Ok "$Name started (pid $($p.Id), log .run\$Name.log)"
    } else {
        Write-Bad "$Name exited immediately — last lines of .run\$Name.err.log:"
        Get-Content (Join-Path $RunDir "$Name.err.log") -Tail 15 -ErrorAction SilentlyContinue |
            ForEach-Object { Write-Host "      $_" }
    }
}

function Stop-Tracked {
    param($Name)
    $procId = Get-TrackedPid $Name
    if (-not $procId) { return }

    # `npm run dev` becomes node, and a solo worker still spawns helpers.
    # Children have to be collected before the parent dies, or they are
    # reparented and there is no way left to find them.
    $tree = @()
    function Get-Children { param($ParentId)
        Get-CimInstance Win32_Process -Filter "ParentProcessId=$ParentId" -ErrorAction SilentlyContinue |
            ForEach-Object { Get-Children $_.ProcessId; $_.ProcessId }
    }
    try { $tree = @(Get-Children $procId) } catch { $tree = @() }
    $tree += $procId

    foreach ($t in $tree) {
        Stop-Process -Id $t -Force -ErrorAction SilentlyContinue
    }
    Remove-Item (Join-Path $RunDir "$Name.pid") -ErrorAction SilentlyContinue
    Write-Ok "$Name stopped"
}

# ─── early-exit modes ────────────────────────────────────────────────────────

if ($Stop -or $Restart) {
    Write-Step 'Stopping'
    foreach ($n in @('web','worker','api')) { Stop-Tracked $n }
    Write-Info 'Redis left running — it is a service, not ours to stop.'
    if ($Stop) { exit 0 }
}

if ($Status) {
    Write-Step 'Status'
    foreach ($n in @('api','worker','web')) {
        if (Test-Tracked $n) { Write-Ok "$n running (pid $(Get-TrackedPid $n))" }
        else { Write-Info "$n not running" }
    }
    if (Test-Port 6379) { Write-Ok 'redis responding' } else { Write-Info 'redis not responding' }
    if (Test-Port 8000) {
        try { Write-Host "`n  $((Invoke-WebRequest 'http://localhost:8000/api/health' -UseBasicParsing -TimeoutSec 5).Content)`n" }
        catch { }
    }
    exit 0
}

if ($Logs) {
    $f = Join-Path $RunDir 'api.log'
    if (-not (Test-Path $f)) { Write-Bad 'No logs yet. Run .\scripts\start.ps1 first.'; exit 1 }
    Write-Info 'Following api.log — worker.log and web.log are alongside it.'
    Get-Content $f -Wait -Tail 40
    exit 0
}

# ═════════════════════════════════════════════════════════════════════════════
Write-Host "Deep Truth" -ForegroundColor Cyan -NoNewline
Write-Host " - $RepoRoot" -ForegroundColor DarkGray
$missing = 0

Write-Step '1. Toolchain'

$Py = $null
foreach ($cand in @('python', 'py')) {
    $exe = Get-Command $cand -ErrorAction SilentlyContinue
    if (-not $exe) { continue }
    try {
        $v = & $cand -c "import sys; print('%d.%d' % sys.version_info[:2])" 2>$null
        if ($v -and [version]$v -ge [version]'3.10') { $Py = $cand; break }
    } catch { }
}
if ($Py) { Write-Ok "python $(& $Py -c 'import platform;print(platform.python_version())') ($Py)" }
else {
    Write-Bad 'Python 3.10+ not found'
    Write-Info 'Install from https://python.org/downloads (tick "Add python.exe to PATH")'
    Write-Info 'or:  winget install Python.Python.3.12'
    $missing = 1
}

if (Get-Command node -ErrorAction SilentlyContinue) { Write-Ok "node $(node --version)" }
elseif (-not $NoWeb) {
    Write-Bad 'node not found (needed for the console; -NoWeb skips it)'
    Write-Info 'winget install OpenJS.NodeJS.LTS'
    $missing = 1
}

if (Get-Command ffmpeg -ErrorAction SilentlyContinue) { Write-Ok 'ffmpeg present' }
else { Write-Warn 'ffmpeg not on PATH — PyAV bundles its own, so video usually still works' }

# ─── 2. Redis ────────────────────────────────────────────────────────────────
Write-Step '2. Redis'

if (Test-Port 6379) {
    Write-Ok 'redis responding on 6379'
} else {
    Write-Bad 'nothing listening on 6379 — the queue and cache cannot run without it'
    $svc = Get-Service -Name 'Memurai' -ErrorAction SilentlyContinue
    if ($svc) {
        Write-Info 'Memurai is installed but stopped.'
        if (-not $Check -and (Confirm-Action 'Start the Memurai service?')) {
            Start-Service Memurai; Start-Sleep 2
            if (Test-Port 6379) { Write-Ok 'Memurai started' } else { Write-Bad 'still not responding'; $missing = 1 }
        } else { $missing = 1 }
    } else {
        Write-Info 'There is no official Redis for Windows. Pick one:'
        Write-Info '  WSL2    wsl --install; then inside WSL:'
        Write-Info '          sudo apt install redis-server && sudo service redis-server start'
        Write-Info '  Memurai winget install Memurai.MemuraiDeveloper   (native service)'
        Write-Info '  Docker  docker run -d -p 6379:6379 redis:7-alpine'
        Write-Info 'Any of them works unchanged — the app only needs localhost:6379.'
        $missing = 1
    }
}

# ─── 3. Package layout ───────────────────────────────────────────────────────
Write-Step '3. Package layout'

$leaf = Split-Path -Leaf $RepoRoot
$link = Join-Path (Split-Path -Parent $RepoRoot) 'deeptruth_pipeline'
if ($leaf -eq 'deeptruth_pipeline') {
    Write-Ok 'repo directory is already named deeptruth_pipeline'
} elseif (Test-Path $link) {
    Write-Ok "import link present ($link)"
} else {
    # Single-quoted: a backtick is PowerShell's escape character, so the
    # backticks that would read naturally around a code name silently eat the
    # next letter.
    Write-Bad ("import deeptruth_pipeline will fail: this folder is named '" + $leaf + "'")
    if (-not $Check) {
        Write-Info 'Simplest fix: rename the cloned folder to  deeptruth_pipeline'
        Write-Info 'Or, in an admin PowerShell, create a junction:'
        Write-Info "  New-Item -ItemType Junction -Path '$link' -Target '$RepoRoot'"
        try {
            New-Item -ItemType Junction -Path $link -Target $RepoRoot -ErrorAction Stop | Out-Null
            Write-Ok "created junction $link"
        } catch {
            Write-Warn 'Could not create the junction (needs admin or Developer Mode).'
            $missing = 1
        }
    } else { $missing = 1 }
}

# ─── 4. Python dependencies ──────────────────────────────────────────────────
Write-Step '4. Python dependencies'

if ($Py) {
    $probe = @'
import importlib
need = {"torch":"torch","transformers":"transformers","peft":"peft","fastapi":"fastapi",
        "celery":"celery","redis":"redis","httpx":"httpx","reportlab":"reportlab",
        "aiosqlite":"aiosqlite","PIL":"pillow","cv2":"opencv-python","av":"av","uvicorn":"uvicorn"}
missing=[]
for m,p in need.items():
    try: importlib.import_module(m)
    except Exception: missing.append(p)
print(" ".join(sorted(set(missing))))
'@
    $missingPy = (& $Py -c $probe 2>$null)
    if ([string]::IsNullOrWhiteSpace($missingPy)) {
        Write-Ok 'all Python packages present'
    } elseif ($Check) {
        Write-Bad "missing: $missingPy"; $missing = 1
    } else {
        Write-Warn "missing: $missingPy"
        Write-Info 'installing from server\requirements.txt (several minutes)'
        & $Py -m pip install -q -r (Join-Path $ServerDir 'requirements.txt')
        if ($LASTEXITCODE -ne 0) { Write-Bad 'pip install failed'; exit 1 }
        Write-Ok 'Python dependencies installed'
    }
}

# ─── 5. Console dependencies ─────────────────────────────────────────────────
if (-not $NoWeb) {
    Write-Step '5. Console dependencies'
    if (Test-Path (Join-Path $ClientDir 'node_modules')) {
        Write-Ok 'node_modules present'
    } elseif ($Check) {
        Write-Bad 'client\node_modules missing'; $missing = 1
    } else {
        Write-Info 'running npm install (several minutes on a cold cache)'
        Push-Location $ClientDir
        cmd /c "npm install --no-fund --no-audit > `"$RunDir\npm-install.log`" 2>&1"
        Pop-Location
        if (Test-Path (Join-Path $ClientDir 'node_modules')) { Write-Ok 'console dependencies installed' }
        else { Write-Bad 'npm install failed — see .run\npm-install.log'; exit 1 }
    }
}

# ─── 6. Model weights ────────────────────────────────────────────────────────
Write-Step '6. Model weights'

function Count-Adapters { param($Dir)
    if (-not (Test-Path $Dir)) { return 0 }
    return (Get-ChildItem $Dir -Recurse -Depth 1 -Filter 'adapter_config.json' -ErrorAction SilentlyContinue).Count
}
$v = Count-Adapters (Join-Path $RepoRoot 'videos_checkpoints')
$i = Count-Adapters (Join-Path $RepoRoot 'images_checkpoints')
if ($v -gt 0) { Write-Ok "video: $v adapters" } else { Write-Bad 'no video adapters in videos_checkpoints\'; $missing = 1 }
if ($i -gt 0) { Write-Ok "image: $i adapters" } else { Write-Bad 'no image adapters in images_checkpoints\'; $missing = 1 }

if (Get-ChildItem (Join-Path $RepoRoot 'checkpoints\audios_checkpoints') -Filter *.pt -ErrorAction SilentlyContinue) {
    Write-Ok 'audio checkpoint present'
} else { Write-Info 'audio: no .pt — engine stays a stub (expected; weights are gitignored)' }

if (Get-ChildItem (Join-Path $RepoRoot 'checkpoints\srm_checkpoints') -Filter *.pt -ErrorAction SilentlyContinue) {
    Write-Ok 'SRM checkpoint present'
} else { Write-Info 'SRM: no .pt — features only, no verdict (expected)' }

if ($Check) {
    if ($missing -eq 0) { Write-Host "`nEverything needed is present." -ForegroundColor Green }
    else { Write-Host "`nSome things are missing (see [FAIL] above). Run without -Check to install." -ForegroundColor Yellow }
    exit $missing
}
if ($missing -ne 0) { Write-Host "`nCannot start — unresolved problems above." -ForegroundColor Red; exit 1 }

# ─── 7. Start ────────────────────────────────────────────────────────────────
Write-Step '7. Starting services'

if ((Test-Port 8000) -and -not (Test-Tracked 'api')) {
    Write-Warn 'port 8000 already in use by something this script did not start'
} else {
    Start-Tracked 'api' $ServerDir $Py @('main.py')
}

# --pool=solo is not a tuning choice. Celery's default prefork pool needs
# fork(), which Windows does not provide; without this the worker accepts a
# task and then dies or hangs. One task at a time per worker is the cost.
$celery = (Get-Command celery -ErrorAction SilentlyContinue)
$celeryExe  = if ($celery) { 'celery' } else { $Py }
$celeryArgs = if ($celery) { @() } else { @('-m','celery') }
$celeryArgs += @('-A','app.queue.celery_app','worker','--loglevel=info',
                 '--pool=solo','--queues=analysis','--without-gossip','--without-mingle')

for ($n = 1; $n -le $Workers; $n++) {
    Start-Tracked "worker$n" $ServerDir $celeryExe ($celeryArgs + @('--hostname', "w$n@%h"))
}
if ($Workers -gt 0) {
    Write-Info "$Workers solo worker(s): each handles one file at a time, so $Workers run in parallel."
}

if (-not $NoWeb) {
    if ((Test-Port 3000) -and -not (Test-Tracked 'web')) {
        Write-Warn 'port 3000 already in use by something this script did not start'
    } else {
        Start-Tracked 'web' $ClientDir 'cmd.exe' @('/c','npm','run','dev')
    }
}

# ─── 8. Health ───────────────────────────────────────────────────────────────
Write-Step '8. Health'
Write-Host '  waiting for the API' -NoNewline
$health = $null
for ($t = 0; $t -lt 45; $t++) {
    try {
        $health = (Invoke-WebRequest 'http://localhost:8000/api/health' -UseBasicParsing -TimeoutSec 2).Content
        break
    } catch { Write-Host '.' -NoNewline; Start-Sleep -Seconds 1 }
}
Write-Host ''

if (-not $health) {
    Write-Bad 'API not answering — see .run\api.err.log'
} else {
    Write-Ok "API: $health"
    if ($health -match '"status":"ok"') { Write-Ok 'workers connected' }
    else { Write-Warn 'no worker registered yet — wait a moment, then .\scripts\start.ps1 -Status' }
}

Write-Host "`nRunning." -ForegroundColor Green
Write-Host '  Console   http://localhost:3000' -ForegroundColor Cyan
Write-Host '  API docs  http://localhost:8000/api/docs' -ForegroundColor Cyan
Write-Host ''
Write-Host '  logs    .\scripts\start.ps1 -Logs'
Write-Host '  status  .\scripts\start.ps1 -Status'
Write-Host '  stop    .\scripts\start.ps1 -Stop'
Write-Host ''
Write-Host '  Note' -ForegroundColor Yellow -NoNewline
Write-Host ' the worker does not hot-reload. After changing Python code:'
Write-Host '       .\scripts\start.ps1 -Restart'
Write-Host ''
