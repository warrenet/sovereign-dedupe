param(
  [ValidateSet("audit","execute","rollback","selftest")] [string]$cmd = "audit",
  [switch]$Apply,
  [string]$Confirm = ""
)
$py = "python"
if ($cmd -eq "audit")   { & $py .\sovereign_dedupe.py audit --preview 10 }
if ($cmd -eq "selftest"){ & $py .\sovereign_dedupe.py selftest }
if ($cmd -eq "execute") {
  $args = @("execute","--verify-hash")
  if ($Apply) { $args += @("--apply","--confirm",$Confirm) }
  & $py .\sovereign_dedupe.py @args
}
if ($cmd -eq "rollback") {
  if (-not $Apply) { & $py .\sovereign_dedupe.py rollback --manifest .\Sovereign_Staging\sovereign_manifest_*.json }
  else { & $py .\sovereign_dedupe.py rollback --manifest .\Sovereign_Staging\sovereign_manifest_*.json --apply }
}
