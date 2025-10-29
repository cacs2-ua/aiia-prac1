# eval_detection_efficiency.py
# Join perf_client.csv (from grabber with frame_id + server_count) and
# local_counts.csv (from local_count_from_folder.py). Compute MAE/MAPE and
# binary presence metrics (Precision/Recall/F1).
#
# Usage:
#   python eval_detection_efficiency.py --perf-client perf_client.csv --local-counts local_counts.csv
# Options:
#   --out-csv joined_counts.csv --presence-threshold 1

import argparse, csv
from collections import Counter

def read_csv(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))

def safe_int(x, default=None):
    try: return int(x)
    except: return default

def safe_float(x, default=None):
    try: return float(x)
    except: return default

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--perf-client", default="perf_client.csv")   # must contain frame_id, server_count
    ap.add_argument("--local-counts", default="local_counts.csv") # must contain frame_id, local_count
    ap.add_argument("--out-csv", default="joined_counts.csv")
    ap.add_argument("--presence-threshold", type=int, default=1)  # ≥1 person => present
    args = ap.parse_args()

    perf = read_csv(args.perf_client)
    loc  = read_csv(args.local_counts)

    # Map local counts by frame_id
    loc_map = {safe_int(r.get("frame_id")): safe_int(r.get("local_count"), 0) for r in loc}

    rows = []
    for r in perf:
        fid = safe_int(r.get("frame_id"))
        if fid is None: 
            continue
        s_cnt = safe_int(r.get("server_count"))
        l_cnt = loc_map.get(fid)
        if s_cnt is None or l_cnt is None:
            continue
        rows.append({"frame_id": fid, "server_count": s_cnt, "local_count": l_cnt})

    with open(args.out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["frame_id","server_count","local_count"])
        w.writeheader()
        w.writerows(rows)

    if not rows:
        print("No hay filas emparejadas. Revisa que perf_client.csv tenga frame_id y server_count, y local_counts.csv tenga frame_id y local_count.")
        return

    # MAE / MAPE (MAPE solo cuando local_count>0)
    abs_err = [abs(r["server_count"] - r["local_count"]) for r in rows]
    mae = sum(abs_err) / len(abs_err)

    mape_terms = [abs(r["server_count"] - r["local_count"]) / r["local_count"]
                  for r in rows if r["local_count"] and r["local_count"] > 0]
    mape = (sum(mape_terms)/len(mape_terms))*100 if mape_terms else float("nan")

    # Presence/absence
    thr = args.presence_threshold
    y_true = [1 if r["local_count"] >= thr else 0 for r in rows]
    y_pred = [1 if r["server_count"] >= thr else 0 for r in rows]

    cm = Counter()
    for t,p in zip(y_true,y_pred):
        if t==1 and p==1: cm["TP"]+=1
        elif t==1 and p==0: cm["FN"]+=1
        elif t==0 and p==1: cm["FP"]+=1
        else: cm["TN"]+=1

    TP, FP, FN, TN = cm["TP"], cm["FP"], cm["FN"], cm["TN"]
    precision = TP / (TP+FP) if (TP+FP)>0 else float("nan")
    recall    = TP / (TP+FN) if (TP+FN)>0 else float("nan")
    f1        = 2*precision*recall/(precision+recall) if (precision+recall)>0 else float("nan")
    acc       = (TP+TN)/(TP+TN+FP+FN) if (TP+TN+FP+FN)>0 else float("nan")

    print(f"Filas emparejadas: {len(rows)}")
    print(f"MAE (error absoluto medio): {mae:.3f}")
    if mape==mape:
        print(f"MAPE (solo local_count>0): {mape:.2f}%")
    else:
        print("MAPE: no aplicable (no hubo local_count>0)")
    print(f"Umbral de presencia: ≥{thr}")
    print(f"TP={TP}  FP={FP}  FN={FN}  TN={TN}")
    print(f"Precision={precision:.3f}  Recall={recall:.3f}  F1={f1:.3f}  Accuracy={acc:.3f}")

if __name__ == "__main__":
    main()
