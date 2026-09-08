$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
Add-Type -AssemblyName System.Windows.Forms

$base = Join-Path (Get-Location) 'WeatherData_AWS\interpolation_batch\fullnet\elevation_hour_effect'
$metricsPath = Join-Path $base 'standard_metrics_station_hour.csv'
$cohortPath = Join-Path $base 'station_cohort.csv'
$outPath = Join-Path $base 'plots\10_elevation_vs_mae_improvement.png'
$summaryPath = Join-Path $base 'elevation_vs_mae_improvement_hourly.csv'

$cohort = @{}
foreach ($r in (Import-Csv -LiteralPath $cohortPath)) {
    if ($r.COHORT_MAIN -match '^(?i:true)$') { $cohort[[string]$r.STN] = $true }
}

$pairs = @{}
foreach ($r in (Import-Csv -LiteralPath $metricsPath)) {
    $stn = [string]$r.STN
    if (-not $cohort.ContainsKey($stn)) { continue }
    $key = "$stn|$($r.HOUR)"
    if (-not $pairs.ContainsKey($key)) {
        $pairs[$key] = [ordered]@{STN=$stn; Name=$r.'지점명'; Elev=[double]$r.elev_m; Hour=[int]$r.HOUR; None=$null; Fixed=$null}
    }
    if ($r.METHOD -eq 'none') { $pairs[$key].None = [double]$r.MAE_C }
    if ($r.METHOD -eq 'fixed_lapse') { $pairs[$key].Fixed = [double]$r.MAE_C }
}

$rows = @()
foreach ($p in $pairs.Values) {
    if ($null -ne $p.None -and $null -ne $p.Fixed) {
        $rows += [pscustomobject]@{ STN=$p.STN; Name=$p.Name; Elev=$p.Elev; Hour=$p.Hour; Improvement=($p.None-$p.Fixed) }
    }
}

function Get-Regression($data) {
    $n = $data.Count
    $xbar = ($data | Measure-Object Elev -Average).Average
    $ybar = ($data | Measure-Object Improvement -Average).Average
    $num = 0.0; $den = 0.0
    foreach ($d in $data) { $num += ($d.Elev-$xbar)*($d.Improvement-$ybar); $den += ($d.Elev-$xbar)*($d.Elev-$xbar) }
    $slope = if ($den -eq 0) { 0 } else { $num/$den }
    [pscustomobject]@{ Slope=$slope; Intercept=($ybar-$slope*$xbar); N=$n }
}

$hourly = @()
foreach ($h in 0..23) {
    $d = @($rows | Where-Object Hour -eq $h)
    if ($d.Count -gt 1) {
        $fit = Get-Regression $d
        $hourly += [pscustomobject]@{Hour=$h; Slope_C_per_100m=($fit.Slope*100); N=$fit.N}
    }
}
$hourly | Export-Csv -LiteralPath $summaryPath -NoTypeInformation -Encoding utf8

$W=1800; $H=1200
$bmp = New-Object System.Drawing.Bitmap($W,$H)
$g = [System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode = [System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.Clear([System.Drawing.Color]::White)
$fontTitle = New-Object System.Drawing.Font('Malgun Gothic',18,[System.Drawing.FontStyle]::Bold)
$fontPanel = New-Object System.Drawing.Font('Malgun Gothic',15,[System.Drawing.FontStyle]::Bold)
$font = New-Object System.Drawing.Font('Malgun Gothic',11)
$fontSmall = New-Object System.Drawing.Font('Malgun Gothic',9)
$penAxis = New-Object System.Drawing.Pen([System.Drawing.Color]::Black,1.4)
$penGrid = New-Object System.Drawing.Pen([System.Drawing.Color]::LightGray,1)
$penFit = New-Object System.Drawing.Pen([System.Drawing.Color]::Black,2)
$penHour = New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(106,81,163),3)
$brushPos = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(43,140,190))
$brushNeg = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(217,95,14))
$brushText = New-Object System.Drawing.SolidBrush([System.Drawing.Color]::Black)

function Draw-String($txt,$x,$y,$f=$font) { $g.DrawString([string]$txt,$f,$brushText,[float]$x,[float]$y) }
function Plot-Panel($x0,$y0,$pw,$ph,$hour) {
    $d = @($rows | Where-Object Hour -eq $hour)
    $fit = Get-Regression $d
    $left=$x0+80; $right=$x0+$pw-20; $top=$y0+45; $bottom=$y0+$ph-65
    $xmin=0; $xmax=1750; $ys=$d | ForEach-Object Improvement
    $ymin=[math]::Floor((($ys | Measure-Object -Minimum).Minimum-0.15)*10)/10
    $ymax=[math]::Ceiling((($ys | Measure-Object -Maximum).Maximum+0.15)*10)/10
    if ($ymin -ge 0) {$ymin=-0.2}; if ($ymax -le 0) {$ymax=0.2}
    function X($v){ return $left + ($v-$xmin)/($xmax-$xmin)*($right-$left) }
    function Y($v){ return $bottom - ($v-$ymin)/($ymax-$ymin)*($bottom-$top) }
    for ($tick=0; $tick -le 1500; $tick+=500) { $xx=X $tick; $g.DrawLine($penGrid,$xx,$top,$xx,$bottom); Draw-String "$tick" ($xx-12) ($bottom+8) $fontSmall }
    $step=0.2; for ($t=$ymin; $t -le $ymax+0.001; $t+=$step) { $yy=Y $t; $g.DrawLine($penGrid,$left,$yy,$right,$yy); Draw-String (("{0:N1}" -f $t)) ($left-48) ($yy-7) $fontSmall }
    $g.DrawLine($penAxis,$left,$top,$left,$bottom); $g.DrawLine($penAxis,$left,$bottom,$right,$bottom)
    $yzero=Y 0; $g.DrawLine((New-Object System.Drawing.Pen([System.Drawing.Color]::Gray,1.2)),$left,$yzero,$right,$yzero)
    foreach ($p in $d) { $xx=X $p.Elev; $yy=Y $p.Improvement; $b=if($p.Improvement -ge 0){$brushPos}else{$brushNeg}; $g.FillEllipse($b,$xx-3,$yy-3,6,6) }
    $x1=$xmin; $x2=$xmax; $g.DrawLine($penFit,(X $x1),(Y ($fit.Intercept+$fit.Slope*$x1)),(X $x2),(Y ($fit.Intercept+$fit.Slope*$x2)))
    Draw-String ("{0:00}시" -f $hour) ($x0+10) ($y0+8) $fontPanel
    Draw-String (("기울기: {0:+0.000;-0.000;0.000} °C/100 m" -f ($fit.Slope*100))) ($x0+10) ($y0+$ph-38) $fontSmall
    Draw-String '관측소 해발고도 (m)' ($x0+($pw/2)-75) ($y0+$ph-28) $fontSmall
    $g.TranslateTransform($x0+17,$y0+$ph/2+35); $g.RotateTransform(-90); Draw-String 'MAE 개선량 (보정 전 − 보정 후, °C)' 0 0 $fontSmall; $g.ResetTransform()
}

Draw-String '관측소 고도와 고도보정에 따른 MAE 개선량의 관계' 490 12 $fontTitle
Plot-Panel 30 55 850 480 6
Plot-Panel 920 55 850 480 15

$x0=100; $y0=610; $pw=1600; $ph=460; $left=$x0+70; $right=$x0+$pw-20; $top=$y0+45; $bottom=$y0+$ph-65
$svals=@($hourly.Slope_C_per_100m); $ymin=[math]::Floor((($svals | Measure-Object -Minimum).Minimum-0.01)*100)/100; $ymax=[math]::Ceiling((($svals | Measure-Object -Maximum).Maximum+0.01)*100)/100
function HX($v){return $left+$v/23*($right-$left)}; function HY($v){return $bottom-($v-$ymin)/($ymax-$ymin)*($bottom-$top)}
for($h=0;$h -le 23;$h++){ $xx=HX $h; $g.DrawLine($penGrid,$xx,$top,$xx,$bottom); Draw-String "$h" ($xx-5) ($bottom+8) $fontSmall }
$step=0.02; for($t=$ymin;$t -le $ymax+0.001;$t+=$step){$yy=HY $t; $g.DrawLine($penGrid,$left,$yy,$right,$yy); Draw-String (("{0:N2}" -f $t)) ($left-50) ($yy-7) $fontSmall}
$g.DrawLine($penAxis,$left,$top,$left,$bottom); $g.DrawLine($penAxis,$left,$bottom,$right,$bottom); $g.DrawLine((New-Object System.Drawing.Pen([System.Drawing.Color]::Gray,1.2)),$left,(HY 0),$right,(HY 0))
$pts=New-Object System.Drawing.PointF[] $hourly.Count; $idx=0; foreach($r in $hourly){$pts[$idx]=New-Object System.Drawing.PointF((HX $r.Hour),(HY $r.Slope_C_per_100m));$idx++}; if($pts.Count -gt 1){$g.DrawLines($penHour,$pts)}; foreach($pt in $pts){$g.FillEllipse((New-Object System.Drawing.SolidBrush([System.Drawing.Color]::FromArgb(106,81,163))),$pt.X-4,$pt.Y-4,8,8)}
Draw-String '시간대별: 고도가 높아질수록 고도보정 개선량이 얼마나 달라지는가' 540 625 $fontPanel
Draw-String '시각 (KST)' 860 1045 $fontSmall
$g.TranslateTransform(35,870); $g.RotateTransform(-90); Draw-String '고도 100 m 증가 시 MAE 개선량 변화 (°C)' 0 0 $fontSmall; $g.ResetTransform()
Draw-String '양수: 고도보정 후 MAE 감소(개선)  |  음수: MAE 증가(악화)  |  회귀선은 평균적인 경향' 510 1145 $fontSmall

$bmp.Save($outPath,[System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose(); $bmp.Dispose()
Write-Output $outPath
$hourly | Format-Table -AutoSize
