<#
.SYNOPSIS
  One-time setup so the USB cameras appear inside WSL automatically at logon.

.DESCRIPTION
  usbipd-win needs two steps per camera:

    bind    marks the device shareable. PERSISTS across reboots. Needs admin.
    attach  hands it to WSL.           DOES NOT persist. No admin needed (usbipd 4.x).

  That second step is what you have been doing by hand every boot. This script does
  the bind once, then registers a per-camera logon task that runs

      usbipd attach --wsl --busid <id> --auto-attach

  --auto-attach keeps running and re-attaches whenever the camera is replugged or WSL
  restarts, so the device survives more than just the first boot. Each task also pokes
  WSL awake first, because attach needs a running distro to hand the device to.

  Net effect: power on -> log in -> open WSL -> /dev/video* is already there.

.PARAMETER BusId
  One or more BUSIDs from `usbipd list`, e.g. -BusId 1-4,1-8

.PARAMETER HardwareId
  Alternative to -BusId: VID:PID, e.g. -HardwareId 046d:0825
  More robust than a BUSID (which changes if you move the camera to another port),
  but useless if both cameras are the same model — they share a hardware id.
  Use -BusId in that case.

.PARAMETER Distro
  WSL distro to attach to. Defaults to the WSL default distro.

.PARAMETER NoStartWsl
  Don't start WSL at logon. Attach will then only succeed once you open WSL yourself.

.PARAMETER Remove
  Undo: delete the scheduled tasks (does not unbind the devices).

.EXAMPLE
  # in an Administrator PowerShell, from the repo root:
  usbipd list                                          # find the two cameras
  .\scripts\setup_cameras_windows.ps1 -BusId 1-4,1-8

.EXAMPLE
  .\scripts\setup_cameras_windows.ps1 -Remove
#>
[CmdletBinding(DefaultParameterSetName = 'ByBusId')]
param(
    [Parameter(ParameterSetName = 'ByBusId')]
    [string[]]$BusId,

    [Parameter(ParameterSetName = 'ByHardwareId')]
    [string[]]$HardwareId,

    [string]$Distro = '',
    [switch]$NoStartWsl,
    [switch]$Remove
)

$ErrorActionPreference = 'Stop'
$TaskPrefix = 'robot_control-usbipd-'

function Test-Admin {
    $id = [Security.Principal.WindowsIdentity]::GetCurrent()
    (New-Object Security.Principal.WindowsPrincipal $id).IsInRole(
        [Security.Principal.WindowsBuiltInRole]::Administrator)
}

# ---------------------------------------------------------------- remove mode
if ($Remove) {
    $tasks = Get-ScheduledTask | Where-Object { $_.TaskName -like "$TaskPrefix*" }
    if (-not $tasks) { Write-Host 'Nothing to remove.'; exit 0 }
    foreach ($t in $tasks) {
        Unregister-ScheduledTask -TaskName $t.TaskName -Confirm:$false
        Write-Host "removed task $($t.TaskName)"
    }
    Write-Host ''
    Write-Host 'Devices are still bound. To fully undo, run as admin:' -ForegroundColor Yellow
    Write-Host '    usbipd unbind --all'
    exit 0
}

# ---------------------------------------------------------------- preflight
if (-not (Get-Command usbipd -ErrorAction SilentlyContinue)) {
    Write-Error 'usbipd not found. Install it first:  winget install usbipd'
}

$devices = @()
if ($BusId)      { $devices = $BusId      | ForEach-Object { @{ Flag = '--busid';       Value = $_ } } }
if ($HardwareId) { $devices = $HardwareId | ForEach-Object { @{ Flag = '--hardware-id'; Value = $_ } } }

if (-not $devices) {
    Write-Host ''
    Write-Host 'No device specified. Here is what usbipd can see:' -ForegroundColor Cyan
    Write-Host ''
    usbipd list
    Write-Host ''
    Write-Host 'Pick the two cameras, then re-run with their BUSIDs, e.g.:' -ForegroundColor Cyan
    Write-Host '    .\scripts\setup_cameras_windows.ps1 -BusId 1-4,1-8'
    exit 1
}

if (-not (Test-Admin)) {
    Write-Error 'Run this in an Administrator PowerShell — `usbipd bind` requires it. (The logon tasks it creates do not.)'
}

# ---------------------------------------------------------------- bind + task
# Keep the woken distro and the attach target the same, or you can end up starting one
# distro and handing the camera to another. Bare `--wsl` means "the default distro".
$wakeArgs   = if ($Distro) { "-d $Distro" } else { '' }
$attachWsl  = if ($Distro) { "--wsl $Distro" } else { '--wsl' }

foreach ($dev in $devices) {
    $flag  = $dev.Flag
    $value = $dev.Value
    $safe  = $value -replace '[^A-Za-z0-9]', '_'
    $task  = "$TaskPrefix$safe"

    Write-Host ""
    Write-Host "=== $value ===" -ForegroundColor Cyan

    # bind is idempotent in practice, but re-binding an already-bound device warns.
    Write-Host "  binding (persists across reboots)..."
    & usbipd bind $flag $value
    if ($LASTEXITCODE -ne 0) {
        Write-Warning "  bind returned $LASTEXITCODE — if it says 'already shared', that is fine."
    }

    # The command the logon task runs. Waking WSL first matters: `attach --wsl` needs a
    # running distro, and at logon there usually isn't one yet.
    $inner = if ($NoStartWsl) {
        "usbipd attach $attachWsl $flag $value --auto-attach"
    } else {
        "wsl.exe $wakeArgs -e true 2>`$null; usbipd attach $attachWsl $flag $value --auto-attach"
    }
    $argument = "-NoProfile -NonInteractive -WindowStyle Hidden -Command `"$inner`""

    $action  = New-ScheduledTaskAction -Execute 'powershell.exe' -Argument $argument
    $trigger = New-ScheduledTaskTrigger -AtLogOn -User $env:USERNAME
    $trigger.Delay = 'PT15S'      # let USB enumeration and the network settle first

    # ExecutionTimeLimit 0 = never kill it. --auto-attach is a long-running watcher,
    # and the default 3-day limit would silently stop it on a machine left running.
    $settings = New-ScheduledTaskSettingsSet `
        -AllowStartIfOnBatteries -DontStopIfGoingOnBatteries `
        -ExecutionTimeLimit ([TimeSpan]::Zero) `
        -RestartCount 3 -RestartInterval (New-TimeSpan -Minutes 1) `
        -MultipleInstances IgnoreNew

    if (Get-ScheduledTask -TaskName $task -ErrorAction SilentlyContinue) {
        Unregister-ScheduledTask -TaskName $task -Confirm:$false
    }
    Register-ScheduledTask -TaskName $task -Action $action -Trigger $trigger `
        -Settings $settings -RunLevel Limited -Force | Out-Null

    Write-Host "  registered logon task: $task"
}

# ---------------------------------------------------------------- report
Write-Host ''
Write-Host 'Done. Verify without rebooting:' -ForegroundColor Green
Write-Host ''
foreach ($dev in $devices) {
    Write-Host "    Start-ScheduledTask -TaskName '$TaskPrefix$($dev.Value -replace '[^A-Za-z0-9]', '_')'"
}
Write-Host ''
Write-Host '    usbipd list                 # STATE should read "Attached"'
Write-Host '    wsl -e ls -l /dev/video*    # the cameras, seen from WSL'
Write-Host ''
Write-Host 'Then, inside WSL:' -ForegroundColor Green
Write-Host '    uv run scripts/verify_cameras.py --save'
Write-Host ''
Write-Host "To undo:  .\scripts\setup_cameras_windows.ps1 -Remove"
