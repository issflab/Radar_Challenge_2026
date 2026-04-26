import re
from pathlib import Path

src = Path("scores/surya_model/Large_Radar_Eval.rtf")
dst = Path("scores/surya_model/Large_Radar_Eval.tsv")

pattern = re.compile(r"(RADAR2026-EVAL\d+)\.flac\s+([-\d.eE+]+)")

with open(src) as fin, open(dst, "w") as fout:
    fout.write("filename\tscore\n")
    count = 0
    for line in fin:
        m = pattern.search(line)
        if m:
            utt_id, score = m.group(1), float(m.group(2))
            fout.write(f"{utt_id}\t{-score}\n")  # negate: bonafide→fake score
            count += 1

print(f"Written {count} rows → {dst}")
