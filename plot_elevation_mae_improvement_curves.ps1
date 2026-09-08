$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing

$base = Join-Path (Get-Location) 'WeatherData_AWS\interpolation_batch\fullnet\elevation_hour_effect'
$metricsPath = Join-Path $base 'standard_metrics_station_hour.csv'
$cohortPath = Join-Path $base 'station_cohort.csv'
$outPath = Join-Path $base 'plots\12_elevation_mae_improvement_8hours.png'

$cohort = @{}
foreach ($r in (Import-Csv -LiteralPath $cohortPath)) {
    if ($r.COHORT_MAIN -match '^(?i:true)$') { $cohort[[string]$r.STN] = $true }
}
$pairs = @{}
foreach ($r in (Import-Csv -LiteralPath $metricsPath)) {
    $stn=[string]$r.STN; if (-not $cohort.ContainsKey($stn)) {continue}; $key="$stn|$($r.HOUR)"
    if (-not $pairs.ContainsKey($key)) {$pairs[$key]=[ordered]@{Elev=[double]$r.elev_m;Hour=[int]$r.HOUR;None=$null;Fixed=$null}}
    if ($r.METHOD -eq 'none') {$pairs[$key].None=[double]$r.MAE_C}
    if ($r.METHOD -eq 'fixed_lapse') {$pairs[$key].Fixed=[double]$r.MAE_C}
}
$rows=@(); foreach($p in $pairs.Values){if($null -ne $p.None -and $null -ne $p.Fixed){$rows += [pscustomobject]@{Elev=$p.Elev;Hour=$p.Hour;Improvement=($p.None-$p.Fixed)}}}

# Use all station-level values to fit a locally weighted elevation trend per hour.
# A local curve preserves non-linearity without compressing all high stations into one decile point.
function Get-Curve($hour) {
    $d=@($rows | Where-Object Hour -eq $hour)
    $neighborCount=[math]::Max(30,[math]::Ceiling($d.Count*0.25))
    $out=@()
    for($x=0;$x -le 1750;$x+=25){
        $near=@($d | ForEach-Object {[pscustomobject]@{DX=([double]$_.Elev-$x);D=[math]::Abs(([double]$_.Elev-$x));Y=[double]$_.Improvement}} | Sort-Object D | Select-Object -First $neighborCount)
        $dmax=[double]$near[-1].D
        if($dmax -le 0){$dmax=1.0}
        $sw=0.0;$sx=0.0;$sy=0.0;$sxx=0.0;$sxy=0.0
        foreach($q in $near){
            $u=[math]::Min(0.999999,([double]$q.D/$dmax))
            $w=[math]::Pow((1-[math]::Pow($u,3)),3)
            $sw+=$w;$sx+=$w*$q.DX;$sy+=$w*$q.Y;$sxx+=$w*$q.DX*$q.DX;$sxy+=$w*$q.DX*$q.Y
        }
        $den=$sw*$sxx-$sx*$sx
        if([math]::Abs($den) -gt 1e-12){$slope=($sw*$sxy-$sx*$sy)/$den;$yhat=($sy-$slope*$sx)/$sw}else{$yhat=$sy/$sw}
        $out += [pscustomobject]@{X=$x;Y=$yhat}
    }
    return $out
}

$hours=@(0,3,6,9,12,15,18,21); $curves=@{}; foreach($h in $hours){$curves[$h]=@(Get-Curve $h)}
$W=1900; $H=650; $bmp=New-Object System.Drawing.Bitmap($W,$H); $g=[System.Drawing.Graphics]::FromImage($bmp); $g.SmoothingMode=[System.Drawing.Drawing2D.SmoothingMode]::AntiAlias; $g.TextRenderingHint=[System.Drawing.Text.TextRenderingHint]::ClearTypeGridFit; $g.Clear([System.Drawing.Color]::White)
$fontAxis=New-Object System.Drawing.Font('Malgun Gothic',15); $font=New-Object System.Drawing.Font('Malgun Gothic',12); $fontSmall=New-Object System.Drawing.Font('Malgun Gothic',11)
$black=New-Object System.Drawing.SolidBrush([System.Drawing.Color]::Black); $axis=New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(35,35,35),1.5); $grid=New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(225,225,225),1); $zero=New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(80,80,80),1.7)
# Eight clearly separated colours, with the compact line-and-marker style of the reference figure.
$cols=@('#1f77b4','#ff7f0e','#2ca02c','#d62728','#9467bd','#8c564b','#e377c2','#7f7f7f'); $pens=@(); $brushes=@(); foreach($c in $cols){$col=[System.Drawing.ColorTranslator]::FromHtml($c);$pen=New-Object System.Drawing.Pen($col,3);$pen.LineJoin=[System.Drawing.Drawing2D.LineJoin]::Round;$pens += $pen;$brushes += New-Object System.Drawing.SolidBrush($col)}
function DS($txt,$x,$y,$f=$font){$g.DrawString([string]$txt,$f,$black,[float]$x,[float]$y)}
$left=130;$right=1860;$top=40;$bottom=530;$xmin=0;$xmax=1750;$ymin=-0.5;$ymax=3.0
function PX($v){return $left+($v-$xmin)/($xmax-$xmin)*($right-$left)}; function PY($v){return $bottom-($v-$ymin)/($ymax-$ymin)*($bottom-$top)}
for($x=0;$x -le 1750;$x+=250){$xx=PX $x;$g.DrawLine($grid,$xx,$top,$xx,$bottom);DS "$x" ($xx-17) ($bottom+10) $fontSmall}
for($i=-1;$i -le 6;$i++){$y=$i*0.5;$yy=PY $y;$g.DrawLine($grid,$left,$yy,$right,$yy);DS (("{0:N1}" -f $y)) ($left-53) ($yy-10) $fontSmall}
$g.DrawRectangle($axis,$left,$top,$right-$left,$bottom-$top);$g.DrawLine($zero,$left,(PY 0),$right,(PY 0))
$idx=0;foreach($h in $hours){$d=$curves[$h];$pts=New-Object System.Drawing.PointF[] $d.Count;$k=0;foreach($p in $d){$pts[$k]=New-Object System.Drawing.PointF((PX $p.X),(PY $p.Y));$k++};if($pts.Count -gt 1){$g.DrawLines($pens[$idx],$pts)};for($k=0;$k -lt $pts.Count;$k+=4){$pt=$pts[$k];$g.FillEllipse($brushes[$idx],$pt.X-3.5,$pt.Y-3.5,7,7)};$idx++}
DS '관측소 해발고도 (m)' 825 590 $fontAxis
$g.TranslateTransform(35,410);$g.RotateTransform(-90);DS 'MAE 개선량 (보정 전 − 보정 후, °C)' 0 0 $fontAxis;$g.ResetTransform()
# Two-row legend in the upper-left, matching the reference figure's compact in-panel legend.
$legendX=155;$legendY=58;$idx=0;foreach($h in $hours){$col=$idx%4;$row=[math]::Floor($idx/4);$xx=$legendX+$col*190;$yy=$legendY+$row*32;$g.DrawLine($pens[$idx],$xx,$yy+8,$xx+34,$yy+8);$g.FillEllipse($brushes[$idx],$xx+14,$yy+4,8,8);DS (("{0:00}시" -f $h)) ($xx+44) ($yy-4) $font;$idx++}
$bmp.Save($outPath,[System.Drawing.Imaging.ImageFormat]::Png);$g.Dispose();$bmp.Dispose();Write-Output $outPath
