"""Exercise 2-6 answer: flag TI reel/tube variants (example data)."""
import runpy, contextlib, io
with contextlib.redirect_stdout(io.StringIO()):
    ns = runpy.run_path("bom_tools.py")
bom = ns["bom"]
# Flag TI rows whose MPNs differ only by a trailing 'R' (reel vs tube)
ti = bom[(bom.manufacturer_norm == "Texas Instruments")
         & (bom.mpn_norm != "") & ~bom.dnp]
mpns = set(ti.mpn_norm)
variant_lines = []
for _, r in ti.iterrows():
    m = r.mpn_norm
    if (m.endswith("R") and m[:-1] in mpns) or (m + "R") in mpns:
        variant_lines.append(r.Line)
bom.loc[bom.Line.isin(variant_lines), "flags"] += ";packaging_variant"
print("packaging_variant lines:", variant_lines)
