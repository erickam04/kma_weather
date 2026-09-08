$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$base = Join-Path (Get-Location) 'WeatherData_AWS\interpolation_batch\fullnet\elevation_hour_effect'
$metricsPath = Join-Path $base 'standard_metrics_station_hour.csv'
$cohortPath = Join-Path $base 'station_cohort.csv'
$csvPath = Join-Path $base 'hourly_mae_improvement_4bands.csv'
$outPath = Join-Path $base 'plots\14_hourly_mae_improvement_4bands.png'

$cohort = @{}
foreach ($r in (Import-Csv -LiteralPath $cohortPath)) {
    if ($r.COHORT_MAIN -match '^(?i:true)$') { $cohort[[string]$r.STN] = $true }
}

$pairs = @{}
foreach ($r in (Import-Csv -LiteralPath $metricsPath)) {
    $stn=[string]$r.STN
    if (-not $cohort.ContainsKey($stn)) { continue }
    $key="$stn|$($r.HOUR)"
    if (-not $pairs.ContainsKey($key)) {
        $pairs[$key]=[ordered]@{STN=$stn;Elev=[double]$r.elev_m;Hour=[int]$r.HOUR;None=$null;Fixed=$null}
    }
    if ($r.METHOD -eq 'none') {$pairs[$key].None=[double]$r.MAE_C}
    if ($r.METHOD -eq 'fixed_lapse') {$pairs[$key].Fixed=[double]$r.MAE_C}
}

$rows=@()
foreach($p in $pairs.Values){
    if($null -eq $p.None -or $null -eq $p.Fixed){continue}
    if($p.Elev -lt 100){$order=0;$band='0–100m'}
    elseif($p.Elev -lt 300){$order=1;$band='100–300m'}
    elseif($p.Elev -lt 500){$order=2;$band='300–500m'}
    else{$order=3;$band='500m 이상'}
    $rows += [pscustomobject]@{STN=$p.STN;Elev=$p.Elev;Hour=$p.Hour;BandOrder=$order;Band=$band;Improvement=($p.None-$p.Fixed)}
}

function Get-Median([object[]]$values){
    $v=@($values|ForEach-Object{[double]$_}|Sort-Object)
    $mid=[math]::Floor($v.Count/2)
    if($v.Count%2 -eq 1){return $v[$mid]}
    return ($v[$mid-1]+$v[$mid])/2.0
}

$summary=@()
for($order=0;$order -le 3;$order++){
    $bandRows=@($rows|Where-Object BandOrder -eq $order)
    $bandName=[string]$bandRows[0].Band
    $n=@($bandRows|Select-Object -ExpandProperty STN -Unique).Count
    foreach($hour in 0..23){
        $d=@($bandRows|Where-Object Hour -eq $hour)
        $summary += [pscustomobject]@{
            band_order=$order
            elevation_band=$bandName
            station_count=$n
            hour_kst=$hour
            median_mae_improvement_c=(Get-Median @($d.Improvement))
        }
    }
}
$summary|Export-Csv -LiteralPath $csvPath -NoTypeInformation -Encoding utf8BOM

$W=1900;$H=650
$bmp=New-Object System.Drawing.Bitmap($W,$H)
$g=[System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode=[System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint=[System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit
$g.Clear([System.Drawing.Color]::White)

$fontAxis=New-Object System.Drawing.Font('Malgun Gothic',15)
$fontLegend=New-Object System.Drawing.Font('Malgun Gothic',12)
$fontTick=New-Object System.Drawing.Font('Malgun Gothic',11)
$black=New-Object System.Drawing.SolidBrush([System.Drawing.Color]::Black)
$axis=New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(35,35,35),1.5)
$grid=New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(225,225,225),1)
$zero=New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(80,80,80),1.7)
$colors=@('#2c7fb8','#41ab5d','#f28e2b','#d9534f')
$pens=@();$brushes=@()
foreach($hex in $colors){
    $col=[System.Drawing.ColorTranslator]::FromHtml($hex)
    $pen=New-Object System.Drawing.Pen($col,3)
    $pen.LineJoin=[System.Drawing.Drawing2D.LineJoin]::Round
    $pens+=$pen;$brushes+=New-Object System.Drawing.SolidBrush($col)
}
function DS($txt,$x,$y,$f=$fontTick){$g.DrawString([string]$txt,$f,$black,[float]$x,[float]$y)}

$left=130;$right=1860;$top=70;$bottom=530;$xmin=0;$xmax=23;$ymin=-0.5;$ymax=2.5
function PX($v){return $left+($v-$xmin)/($xmax-$xmin)*($right-$left)}
function PY($v){return $bottom-($v-$ymin)/($ymax-$ymin)*($bottom-$top)}

foreach($x in @(0,3,6,9,12,15,18,21,23)){
    $xx=PX $x;$g.DrawLine($grid,$xx,$top,$xx,$bottom);DS "$x" ($xx-8) ($bottom+10) $fontTick
}
for($i=-1;$i -le 5;$i++){
    $y=$i*0.5;$yy=PY $y;$g.DrawLine($grid,$left,$yy,$right,$yy);DS (("{0:N1}" -f $y)) ($left-53) ($yy-10) $fontTick
}
$g.DrawRectangle($axis,$left,$top,$right-$left,$bottom-$top)
$g.DrawLine($zero,$left,(PY 0),$right,(PY 0))

for($order=0;$order -le 3;$order++){
    $d=@($summary|Where-Object band_order -eq $order|Sort-Object hour_kst)
    $pts=New-Object System.Drawing.PointF[] $d.Count
    for($k=0;$k -lt $d.Count;$k++){$pts[$k]=New-Object System.Drawing.PointF((PX ([double]$d[$k].hour_kst)),(PY ([double]$d[$k].median_mae_improvement_c)))}
    $g.DrawLines($pens[$order],$pts)
    foreach($pt in $pts){$g.FillEllipse($brushes[$order],$pt.X-4,$pt.Y-4,8,8)}
}

DS '시간 (KST)' 910 590 $fontAxis
$g.TranslateTransform(34,420);$g.RotateTransform(-90);DS 'MAE 개선량 (보정 전 − 보정 후, °C)' 0 0 $fontAxis;$g.ResetTransform()

$legendX=155;$legendY=18;$cellW=360
for($order=0;$order -le 3;$order++){
    $d=@($summary|Where-Object band_order -eq $order|Select-Object -First 1)
    $xx=$legendX+$order*$cellW
    $g.DrawLine($pens[$order],$xx,$legendY+10,$xx+38,$legendY+10)
    $g.FillEllipse($brushes[$order],$xx+15,$legendY+6,8,8)
    DS (("{0} (n={1})" -f $d.elevation_band,$d.station_count)) ($xx+48) ($legendY-3) $fontLegend
}

$bmp.Save($outPath,[System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose();$bmp.Dispose()
Write-Output $csvPath
Write-Output $outPath
