"""
Run this ONCE from your AgentSynth-Trinity directory:
    python patch_app.py

It patches app.py in-place — no download needed.
"""
import re, sys, os

target = os.path.join(os.path.dirname(__file__), "app.py")
if not os.path.exists(target):
    sys.exit(f"❌ Cannot find {target}")

with open(target) as f:
    src = f.read()

if "# PATCHED_NONE_SAFE_v3" in src:
    print("✅ Already patched — nothing to do.")
    sys.exit(0)

# ── FIX 1: Replace the leaderboard f-string block ──────────────────────────
# Find the for-loop that renders pilot leaderboard rows
old_loop = '''                ew    = live["ensemble_weights"].get(p["model"], "")
                tag   = f"weight {ew:.1%}" if ew else ""
                st.markdown(f"""
                <div class="sc-rank-row {rank}">
                  <span class="sc-rank-num">{icon}</span>
                  <span class="sc-rank-model">{p['model']}</span>
                  <span class="sc-rank-score">
                    W={w_val:.4f} &nbsp;|&nbsp; TSTR={t_val:.1%} &nbsp;|&nbsp; MIA={m_val:.3f}
                  </span>
                  <span style="font-size:.7rem;color:{_TEAL_AA};margin-left:auto">{tag}</span>
                </div>""", unsafe_allow_html=True)'''

new_loop = '''                ew    = live["ensemble_weights"].get(p["model"], 0)
                tag   = f"weight {float(ew):.1%}" if ew else ""
                teal_muted = "rgba(0,180,166,0.67)"
                st.markdown(
                    '<div class="sc-rank-row ' + rank + '">'
                    '<span class="sc-rank-num">' + str(icon) + '</span>'
                    '<span class="sc-rank-model">' + str(p["model"]) + '</span>'
                    '<span class="sc-rank-score">'
                    'W=' + w_str + ' &nbsp;|&nbsp; TSTR=' + t_str + ' &nbsp;|&nbsp; MIA=' + m_str +
                    '</span>'
                    '<span style="font-size:.7rem;color:' + teal_muted + ';margin-left:auto">' + tag + '</span>'
                    '</div>',
                    unsafe_allow_html=True)
# PATCHED_NONE_SAFE_v3'''

# ── FIX 2: Ensure w_val/t_raw/m_raw are extracted safely ───────────────────
old_vals = '''                w_val = p.get("wasserstein", p.get("w", 0))
                t_val = p.get("tstr", 0)
                m_val = p.get("mia",  0)
                rank  = rank_css[i] if i < 3 else ""
                icon  = rank_ico[i] if i < 3 else f"#{i+1}"'''

new_vals = '''                w_val = p.get("wasserstein", p.get("w")) or 0.0
                t_raw = p.get("tstr")
                m_raw = p.get("mia")
                w_str = f"{float(w_val):.4f}"
                t_str = f"{float(t_raw):.1%}" if t_raw is not None else "N/A"
                m_str = f"{float(m_raw):.3f}" if m_raw is not None else "N/A"
                rank  = rank_css[i] if i < 3 else ""
                icon  = rank_ico[i] if i < 3 else f"#{i+1}"'''

patched = src
count = 0

if old_vals in patched:
    patched = patched.replace(old_vals, new_vals)
    count += 1
    print("✅ Fix 1: safe value extraction applied")
elif "w_str = f\"{float(w_val):.4f}\"" in patched:
    print("✅ Fix 1: already applied")
    count += 1
else:
    print("⚠️  Fix 1: pattern not found — may already be patched differently")

if old_loop in patched:
    patched = patched.replace(old_loop, new_loop)
    count += 1
    print("✅ Fix 2: None-safe leaderboard HTML applied")
elif "PATCHED_NONE_SAFE_v3" in patched:
    print("✅ Fix 2: already applied")
    count += 1
else:
    print("⚠️  Fix 2: pattern not found — may already be patched differently")

import ast
try:
    ast.parse(patched)
    print("✅ Syntax OK")
except SyntaxError as e:
    sys.exit(f"❌ Syntax error after patch: {e}")

with open(target, "w") as f:
    f.write(patched)
print(f"✅ app.py patched in-place ({patched.count(chr(10))} lines)")
