$ErrorActionPreference = 'Stop'
Add-Type -AssemblyName System.Drawing
$base = Join-Path $PSScriptRoot 'WeatherData_AWS/interpolation_batch/fullnet/elevation_hour_effect'
$out = Join-Path $base 'median_elevation_cases_20260905'
New-Item -ItemType Directory -Path $out -Force | Out-Null
$cohortPath = Join-Path $base 'station_cohort.csv'
$metricsPath = Join-Path $base 'standard_metrics_station_hour.csv'
$stations = @(Import-Csv $cohortPath | Where-Object COHORT_MAIN -eq 'True' | Sort-Object @{Expression={[double]$_.elev_m}},@{Expression={[int]$_.STN}})
if ($stations.Count -ne 513) {throw '주 코호트 513개 불일치'}
$selection = @(); $membership = @(); $offset = 0
for ($i=0; $i -lt 5; $i++) {
    $size = [int][math]::Floor($stations.Count/5)
    if ($i -lt ($stations.Count%5)) {$size++}
    $group = @($stations[$offset..($offset+$size-1)])
    $mid = [int][math]::Floor($size/2)
    $median = [double]$group[$mid].elev_m
    if ($size%2 -eq 0) {$median = ([double]$group[$mid-1].elev_m+$median)/2}
    $chosen = $group | Sort-Object @{Expression={[math]::Abs([double]$_.elev_m-$median)}},@{Expression={[double]$_.elev_m}},@{Expression={[int]$_.STN}} | Select-Object -First 1
    $selection += [pscustomobject]@{GROUP=($i+1);GROUP_N=$size;GROUP_MIN_M=[double]$group[0].elev_m;GROUP_MAX_M=[double]$group[-1].elev_m;GROUP_MEDIAN_M=$median;STN=$chosen.STN;NAME=$chosen.'지점명';ELEV_M=[double]$chosen.elev_m;COVERAGE_DAYS=$chosen.COVERAGE_DAYS;COVERAGE_MONTHS=$chosen.COVERAGE_MONTHS}
    foreach ($s in $group) {$membership += [pscustomobject]@{GROUP=($i+1);STN=$s.STN;NAME=$s.'지점명';ELEV_M=$s.elev_m;SELECTED=($s.STN -eq $chosen.STN)}}
    $offset += $size
}
$metrics = @(Import-Csv $metricsPath)
$rows = @(); $summary = @()
foreach ($s in $selection) {
    foreach ($h in 0..23) {
        $pair = @($metrics | Where-Object { $_.STN -eq $s.STN -and [int]$_.HOUR -eq $h })
        $before = @($pair | Where-Object METHOD -eq 'none')
        $after = @($pair | Where-Object METHOD -eq 'fixed_lapse')
        if ($pair.Count -ne 2 -or $before.Count -ne 1 -or $after.Count -ne 1) {throw "짝 불일치 $($s.STN) $h"}
        if ($before[0].N_PAIRED_HOURS -ne $after[0].N_PAIRED_HOURS) {throw '표본 수 불일치'}
        if ([double]$before[0].elev_m -ne $s.ELEV_M -or [double]$after[0].elev_m -ne $s.ELEV_M) {throw '고도 불일치'}
        $b=[double]$before[0].MAE_C; $a=[double]$after[0].MAE_C
        if ([double]::IsNaN($b) -or [double]::IsNaN($a) -or [double]::IsInfinity($b) -or [double]::IsInfinity($a) -or $a -lt 0 -or $b -lt 0) {throw 'MAE 유효성 오류'}
        $rows += [pscustomobject]@{GROUP=$s.GROUP;STN=$s.STN;NAME=$s.NAME;ELEV_M=$s.ELEV_M;HOUR_KST=$h;N_PAIRED_HOURS=[int]$before[0].N_PAIRED_HOURS;MAE_BEFORE_C=$b;MAE_AFTER_C=$a;MAE_IMPROVEMENT_C=($b-$a)}
    }
    $d=@($rows | Where-Object STN -eq $s.STN)
    $lo=$d|Sort-Object MAE_IMPROVEMENT_C,HOUR_KST|Select-Object -First 1
    $hi=$d|Sort-Object @{Expression={$_.MAE_IMPROVEMENT_C};Descending=$true},HOUR_KST|Select-Object -First 1
    $h6=$d|Where-Object HOUR_KST -eq 6; $h15=$d|Where-Object HOUR_KST -eq 15
    $summary += [pscustomobject]@{STN=$s.STN;NAME=$s.NAME;ELEV_M=$s.ELEV_M;H06_BEFORE=$h6.MAE_BEFORE_C;H06_AFTER=$h6.MAE_AFTER_C;H06_IMPROVEMENT=$h6.MAE_IMPROVEMENT_C;H15_BEFORE=$h15.MAE_BEFORE_C;H15_AFTER=$h15.MAE_AFTER_C;H15_IMPROVEMENT=$h15.MAE_IMPROVEMENT_C;MIN_HOUR=$lo.HOUR_KST;MIN_C=$lo.MAE_IMPROVEMENT_C;MAX_HOUR=$hi.HOUR_KST;MAX_C=$hi.MAE_IMPROVEMENT_C;RANGE_C=($hi.MAE_IMPROVEMENT_C-$lo.MAE_IMPROVEMENT_C)}
}
if ($rows.Count -ne 120 -or @($selection.STN|Select-Object -Unique).Count -ne 5) {throw '최종 크기 불일치'}
$selection | Export-Csv (Join-Path $out 'selected_stations.csv') -NoTypeInformation -Encoding utf8BOM
$membership | Export-Csv (Join-Path $out 'selection_membership.csv') -NoTypeInformation -Encoding utf8BOM
$rows | Export-Csv (Join-Path $out 'station_hour_mae_improvement.csv') -NoTypeInformation -Encoding utf8BOM
$summary | Export-Csv (Join-Path $out 'station_summary.csv') -NoTypeInformation -Encoding utf8BOM

# 기존 논문 그림의 흰 배경·연회색 격자·얇은 선 스타일을 재사용한다.
$W=1900; $H=760
$bmp=New-Object System.Drawing.Bitmap($W,$H)
$bmp.SetResolution(300,300)
$g=[System.Drawing.Graphics]::FromImage($bmp)
$g.SmoothingMode=[System.Drawing.Drawing2D.SmoothingMode]::AntiAlias
$g.TextRenderingHint=[System.Drawing.Text.TextRenderingHint]::AntiAliasGridFit
$g.Clear([System.Drawing.Color]::White)
$fontAxis=New-Object System.Drawing.Font('Malgun Gothic',28,[System.Drawing.FontStyle]::Regular,[System.Drawing.GraphicsUnit]::Pixel)
$fontLegend=New-Object System.Drawing.Font('Malgun Gothic',25,[System.Drawing.FontStyle]::Regular,[System.Drawing.GraphicsUnit]::Pixel)
$fontTick=New-Object System.Drawing.Font('Malgun Gothic',24,[System.Drawing.FontStyle]::Regular,[System.Drawing.GraphicsUnit]::Pixel)
$black=[System.Drawing.Brushes]::Black
$axis=New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(35,35,35),1.6)
$grid=New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(230,230,230),1)
$zero=New-Object System.Drawing.Pen([System.Drawing.Color]::FromArgb(100,100,100),2)
$colors=@('#2166ac','#4393c3','#4b9b68','#e79435','#c63c44')
$left=160; $right=1860; $top=100; $bottom=635
$ext=$rows | Measure-Object MAE_IMPROVEMENT_C -Minimum -Maximum
$ymin=[math]::Floor(([math]::Min(0.0,[double]$ext.Minimum)-0.04)/0.1)*0.1
$ymax=[math]::Ceiling(([math]::Max(0.0,[double]$ext.Maximum)+0.04)/0.1)*0.1
if ($ymin -gt $ext.Minimum -or $ymax -lt $ext.Maximum) {throw '축 범위가 자료를 포함하지 않음'}
$step=0.1; if (($ymax-$ymin) -gt 1.2) {$step=0.2}; if (($ymax-$ymin) -gt 2.4) {$step=0.5}
function PX($v) {return [float]($left+$v/23*($right-$left))}
function PY($v) {return [float]($bottom-($v-$ymin)/($ymax-$ymin)*($bottom-$top))}
function DS($txt,$x,$y,$f=$fontTick) {$g.DrawString([string]$txt,$f,$black,[float]$x,[float]$y)}
foreach ($h in @(0,3,6,9,12,15,18,21,23)) {$x=PX $h; $g.DrawLine($grid,$x,$top,$x,$bottom); DS $h ($x-10) ($bottom+12)}
for ($v=[math]::Ceiling($ymin/$step)*$step; $v -le ($ymax+1e-8); $v+=$step) {$y=PY $v; $g.DrawLine($grid,$left,$y,$right,$y); DS ('{0:F1}' -f $v) ($left-65) ($y-16)}
$g.DrawRectangle($axis,$left,$top,$right-$left,$bottom-$top)
$g.DrawLine($zero,$left,(PY 0),$right,(PY 0))
for ($i=0;$i -lt 5;$i++) {
    $s=$selection[$i]; $d=@($rows|Where-Object STN -eq $s.STN|Sort-Object HOUR_KST)
    $col=[System.Drawing.ColorTranslator]::FromHtml($colors[$i])
    $pen=New-Object System.Drawing.Pen($col,3.3)
    $brush=New-Object System.Drawing.SolidBrush($col)
    $pts=New-Object System.Drawing.PointF[] 24
    for ($k=0;$k -lt 24;$k++) {$pts[$k]=New-Object System.Drawing.PointF((PX $k),(PY $d[$k].MAE_IMPROVEMENT_C))}
    $g.DrawLines($pen,$pts)
    foreach($pt in $pts){$g.FillEllipse($brush,$pt.X-4.5,$pt.Y-4.5,9,9)}
    $lx=175+$i*335; $ly=43
    $g.DrawLine($pen,$lx,$ly+15,$lx+36,$ly+15)
    $g.FillEllipse($brush,$lx+13,$ly+10.5,9,9)
    DS ('{0} ({1:0.#} m)' -f $s.NAME,$s.ELEV_M) ($lx+46) $ly $fontLegend
    $pen.Dispose();$brush.Dispose()
}
DS '시간 (KST)' 920 702 $fontAxis
$g.TranslateTransform(35,555);$g.RotateTransform(-90)
DS 'MAE 개선량 (보정 전 − 보정 후, °C)' 0 0 $fontAxis
$g.ResetTransform()
$imagePath=Join-Path $out '15_hourly_mae_improvement_median_elevation_stations.png'
$bmp.Save($imagePath,[System.Drawing.Imaging.ImageFormat]::Png)
$g.Dispose();$bmp.Dispose()
$manifest=[ordered]@{COHORT_N=$stations.Count;GROUP_SIZES=@($selection.GROUP_N);SELECTED_N=5;HOURLY_ROWS=$rows.Count;ALL_MAX_ELEV_M=[double]$stations[-1].elev_m;SELECTED_MAX_ELEV_M=($selection.ELEV_M|Measure-Object -Maximum).Maximum;INPUT_SHA256=@((Get-FileHash $cohortPath).Hash,(Get-FileHash $metricsPath).Hash);SELECTION='고도 중앙값 절대차 최소, 동률이면 낮은 고도·작은 지점번호';NOTE='실제 관측소 사례이며 집단 성능 대표성을 보장하지 않음'}
$manifest | ConvertTo-Json -Depth 5 | Out-File (Join-Path $out 'run_summary.json') -Encoding utf8BOM
$selection|Format-Table -AutoSize|Out-String|Write-Output
$summary|ConvertTo-Json|Write-Output
Write-Output $imagePath
