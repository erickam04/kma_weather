$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$base = Join-Path (Get-Location) 'WeatherData_AWS\interpolation_batch\fullnet\elevation_hour_effect'
$metricsPath = Join-Path $base 'standard_metrics_station_hour.csv'
$cohortPath = Join-Path $base 'station_cohort.csv'
$csvPath = Join-Path $base 'hourly_mae_improvement_by_100m.csv'
$outPath = Join-Path $base 'plots\13_hourly_mae_improvement_by_100m.png'

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
        $pairs[$key] = [ordered]@{STN=$stn; Elev=[double]$r.elev_m; Hour=[int]$r.HOUR; None=$null; Fixed=$null}
    }
    if ($r.METHOD -eq 'none') { $pairs[$key].None = [double]$r.MAE_C }
    if ($r.METHOD -eq 'fixed_lapse') { $pairs[$key].Fixed = [double]$r.MAE_C }
}

$rows = @()
foreach ($p in $pairs.Values) {
    if ($null -eq $p.None -or $null -eq $p.Fixed) { continue }
    $lo = [math]::Floor($p.Elev / 100.0) * 100
    $rows += [pscustomobject]@{
        STN = $p.STN
        Elev = $p.Elev
        BandLo = [int]$lo
        Hour = $p.Hour
        Improvement = $p.None - $p.Fixed
    }
}

function Get-Median([object[]]$values) {
    $v = @($values | ForEach-Object {[double]$_} | Sort-Object)
    $mid = [math]::Floor($v.Count / 2)
    if ($v.Count % 2 -eq 1) { return $v[$mid] }
    return ($v[$mid-1] + $v[$mid]) / 2.0
}

$bands = @($rows | Select-Object -ExpandProperty BandLo -Unique | Sort-Object)
$summary = @()
foreach ($band in $bands) {
    $bandRows = @($rows | Where-Object BandLo -eq $band)
    $stationCount = @($bandRows | Select-Object -ExpandProperty STN -Unique).Count
    foreach ($hour in 0..23) {
        $d = @($bandRows | Where-Object Hour -eq $hour)
        if ($d.Count -eq 0) { continue }
        $summary += [pscustomobject]@{
            elevation_band = ("{0}-{1}m" -f $band,($band+100))
            band_low_m = $band
            band_high_m = $band + 100
            station_count = $stationCount
            hour_kst = $hour
            median_mae_improvement_c = Get-Median @($d.Improvement)
        }
    }
}
$summary | Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding utf8BOM

$W=1900; $H=760
$bmp=New-Object System.Drawing.Bitmap($W,$H)
$g=[System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode=[System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint=[System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$g.Clear([System.Drawing.Color]::White)

$fontAxis=New-Object System.Drawing.Font('Malgun Gothic',15)
$fontLegend=New-Object System.Drawing.Font('Malgun Gothic',10)
$fontTick=New-Object System.Drawing.Font('Malgun Gothic',11)
$black=New-Object System.Drawing.SolidBrush([System.Drawing.Color]::Black)
$axis=New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(35,35,35),1.5)
$grid=New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(225,225,225),1)
$zero=New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(80,80,80),1.7)

# Turbo-like ordered palette: low elevation (blue) to high elevation (red).
$palette=@(
    '#30123b','#4145ab','#4675ed','#39a2fc','#1bcfd4','#24eca6','#61fc6c','#a4fc3b','#d8ef28',
    '#f6d746','#febc2a','#fe9b20','#f66b19','#e94410','#c9290a','#9f1506','#7a0403'
)
$pens=@();$brushes=@()
for($i=0;$i -lt $bands.Count;$i++){
    $idx=if($bands.Count -le 1){0}else{[math]::Round($i*($palette.Count-1)/($bands.Count-1))}
    $col=[System.Drawing.ColorTranslator]::FromHtml($palette[$idx])
    $pen=New-Object System.Drawing.Pen($col,2.5)
    $pen.LineJoin=[System.Drawing.Drawing2D.LineJoin]::Round
    $pens += $pen
    $brushes += New-Object System.Drawing.SolidBrush($col)
}
function DS($txt,$x,$y,$f=$fontTick){$g.DrawString([string]$txt,$f,$black,[float]$x,[float]$y)}

$left=130;$right=1860;$top=125;$bottom=625;$xmin=0;$xmax=23;$ymin=-1.0;$ymax=5.0
function PX($v){return $left+($v-$xmin)/($xmax-$xmin)*($right-$left)}
function PY($v){return $bottom-($v-$ymin)/($ymax-$ymin)*($bottom-$top)}

foreach($x in @(0,3,6,9,12,15,18,21,23)){
    $xx=PX $x;$g.DrawLine($grid,$xx,$top,$xx,$bottom);DS "$x" ($xx-8) ($bottom+10) $fontTick
}
for($i=-2;$i -le 10;$i++){
    $y=$i*0.5;$yy=PY $y;$g.DrawLine($grid,$left,$yy,$right,$yy);DS (("{0:N1}" -f $y)) ($left-53) ($yy-10) $fontTick
}
$g.DrawRectangle($axis,$left,$top,$right-$left,$bottom-$top)
$g.DrawLine($zero,$left,(PY 0),$right,(PY 0))

for($i=0;$i -lt $bands.Count;$i++){
    $band=$bands[$i]
    $d=@($summary | Where-Object band_low_m -eq $band | Sort-Object hour_kst)
    $pts=New-Object System.Drawing.PointF[] $d.Count
    for($k=0;$k -lt $d.Count;$k++){
        $pts[$k]=New-Object System.Drawing.PointF((PX ([double]$d[$k].hour_kst)),(PY ([double]$d[$k].median_mae_improvement_c)))
    }
    if($pts.Count -gt 1){$g.DrawLines($pens[$i],$pts)}
    foreach($pt in $pts){$g.FillEllipse($brushes[$i],$pt.X-3.2,$pt.Y-3.2,6.4,6.4)}
}

DS '시간 (KST)' 910 680 $fontAxis
$g.TranslateTransform(34,490);$g.RotateTransform(-90);DS 'MAE 개선량 (보정 전 − 보정 후, °C)' 0 0 $fontAxis;$g.ResetTransform()

# Compact three-row legend above the plot.
$cols=6;$legendX=150;$legendY=20;$cellW=280
for($i=0;$i -lt $bands.Count;$i++){
    $col=$i%$cols;$row=[math]::Floor($i/$cols);$xx=$legendX+$col*$cellW;$yy=$legendY+$row*31
    $g.DrawLine($pens[$i],$xx,$yy+8,$xx+30,$yy+8)
    $g.FillEllipse($brushes[$i],$xx+12,$yy+4,8,8)
    $n=@($rows | Where-Object BandLo -eq $bands[$i] | Select-Object -ExpandProperty STN -Unique).Count
    DS (("{0}–{1}m (n={2})" -f $bands[$i],($bands[$i]+100),$n)) ($xx+39) ($yy-4) $fontLegend
}

$bmp.Save($outPath,[System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose();$bmp.Dispose()
Write-Output $csvPath
Write-Output $outPath
