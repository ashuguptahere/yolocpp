#!/usr/bin/env bash
# Join a fresh 1-epoch training-metrics sweep (scripts/train_metrics_sweep.sh
# output) against the yolocpp baseline columns in docs/data/training.csv and
# emit a delta CSV: new value, baseline value, and (new - baseline) for
# mAP@0.5 / mAP / P / R / F1 / CPU% / RSS / VRAM.
#
# Baseline "Y" = yolocpp's previously-recorded 1-epoch numbers. Variants the
# sweep has but the baseline lacks (yolo1/yolo2/yolo4) are emitted with an
# empty baseline and status "new". The sweep's `yolo3u` maps to baseline
# `yolo3` (same u-form weights).
#
# Usage: scripts/build_train_delta.sh <new_metrics.csv> [baseline.csv] > delta.csv
set -u
NEW="${1:?usage: build_train_delta.sh <new_metrics.csv> [baseline.csv]}"
BASE="${2:-$(cd "$(dirname "$0")/.." && pwd)/docs/data/training.csv}"
[ -f "$NEW" ]  || { echo "[error] no new metrics csv: $NEW"  >&2; exit 1; }
[ -f "$BASE" ] || { echo "[error] no baseline csv: $BASE"   >&2; exit 1; }

awk -F, -v OFS=, '
  function idx(hdr, name,   i) { for (i=1;i<=NF;i++) if (hdr[i]==name) return i; return 0 }
  function d(a,b) { if (a=="" || b=="") return ""; return sprintf("%+.4f", a-b) }
  function di(a,b){ if (a=="" || b=="") return ""; return sprintf("%+d",   a-b) }
  function df(a,b){ if (a=="" || b=="") return ""; return sprintf("%+.1f", a-b) }
  # ---- pass 1: baseline ----
  NR==FNR {
    if (FNR==1) { for (i=1;i<=NF;i++) bh[i]=$i;
      c_v=idx(bh,"variant"); c_m50=idx(bh,"1ep_Y_mAP50"); c_m=idx(bh,"1ep_Y_mAP");
      c_p=idx(bh,"1ep_Y_P"); c_r=idx(bh,"1ep_Y_R"); c_f=idx(bh,"1ep_Y_F1");
      c_cpu=idx(bh,"Y_train_CPU_pct"); c_rss=idx(bh,"Y_train_RSS_GB"); c_vram=idx(bh,"Y_train_VRAM_MB");
      next }
    v=$c_v;
    b_m50[v]=$c_m50; b_m[v]=$c_m; b_p[v]=$c_p; b_r[v]=$c_r; b_f[v]=$c_f;
    b_cpu[v]=$c_cpu; b_rss[v]=$c_rss; b_vram[v]=$c_vram;
    next
  }
  # ---- pass 2: new sweep ----
  FNR==1 {
    print "variant",\
      "new_mAP50","base_mAP50","d_mAP50",\
      "new_mAP","base_mAP","d_mAP",\
      "new_P","base_P","d_P",\
      "new_R","base_R","d_R",\
      "new_F1","base_F1","d_F1",\
      "new_CPU_pct","base_CPU_pct","d_CPU",\
      "new_RSS_GB","base_RSS_GB","d_RSS",\
      "new_VRAM_MB","base_VRAM_MB","d_VRAM",\
      "batch","train_sec","status";
    next
  }
  {
    name=$1; key=name; if (name=="yolo3u") key="yolo3";
    nm50=$2; nm=$3; np=$4; nr=$5; nf=$6; ncpu=$7; nrss=$8; nvram=$9;
    nbatch=$10; nsec=$11; nstat=$12;
    has = (key in b_m50);
    bm50 = has? b_m50[key]:""; bm = has? b_m[key]:""; bp = has? b_p[key]:"";
    br = has? b_r[key]:""; bf = has? b_f[key]:"";
    bcpu = has? b_cpu[key]:""; brss = has? b_rss[key]:""; bvram = has? b_vram[key]:"";
    st = nstat; if (!has && nstat=="ok") st="new";
    print name,\
      nm50,bm50,d(nm50,bm50),\
      nm,bm,d(nm,bm),\
      np,bp,d(np,bp),\
      nr,br,d(nr,br),\
      nf,bf,d(nf,bf),\
      ncpu,bcpu,di(ncpu,bcpu),\
      nrss,brss,df(nrss,brss),\
      nvram,bvram,di(nvram,bvram),\
      nbatch,nsec,st
  }
' "$BASE" "$NEW"
