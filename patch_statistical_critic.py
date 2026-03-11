"""
Run from AgentSynth-Trinity root:
    python patch_statistical_critic.py

Patches statistical_critic.py to use normalised Wasserstein
regardless of which version is on disk.
"""
import os, sys, ast, re

target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "agents", "statistical_critic.py")
if not os.path.exists(target):
    # try root level
    target = os.path.join(os.path.dirname(os.path.abspath(__file__)), "statistical_critic.py")
if not os.path.exists(target):
    sys.exit("❌ Cannot find statistical_critic.py in agents/ or root")

with open(target) as f:
    src = f.read()

print(f"Found: {target}")
print(f"  Has _wasserstein_numeric_only: {'_wasserstein_numeric_only' in src}")
print(f"  Has calculate_average_wasserstein_distance: {'calculate_average_wasserstein_distance' in src}")
print(f"  Already patched: {'PATCHED_SC_WASS' in src}")

if "PATCHED_SC_WASS" in src:
    print("✅ Already patched")
    sys.exit(0)

# ── Strategy: replace whatever internal wasserstein method exists ──────────

# Case 1: has _wasserstein_numeric_only (document 1 version)
if "_wasserstein_numeric_only" in src:
    # Find the method and replace it entirely
    start = src.find("    def _wasserstein_numeric_only(")
    # find next method
    end = src.find("\n    def train_and_evaluate(", start)
    if start == -1 or end == -1:
        sys.exit("❌ Could not locate _wasserstein_numeric_only boundaries")

    NEW_METHOD = '''    # PATCHED_SC_WASS
    _ID_KW = ("_id","id_","patient_id","record_id","index","uuid","key","rownum","row_num","seq")

    def _wasserstein_numeric_only(self, real_df, synth_df):
        """Normalised Wasserstein over non-ID numeric columns only."""
        from scipy.stats import wasserstein_distance

        def _skip(col, vals):
            lo = col.lower()
            if lo == "id" or any(lo == k or lo.endswith(k) or lo.startswith(k)
                                 for k in self._ID_KW):
                return True
            if len(vals) > 20 and (len(np.unique(vals)) / len(vals)) > 0.95:
                return True
            return False

        common = [c for c in real_df.columns if c in synth_df.columns]
        real_num  = real_df[common].select_dtypes(include=[np.number])
        synth_num = synth_df[common].select_dtypes(include=[np.number])
        if real_num.empty:
            return float('inf')

        scores = []
        for col in real_num.columns:
            r = real_num[col].dropna().values
            s = synth_num[col].dropna().values
            if len(r) < 2 or len(s) < 2 or _skip(col, r):
                continue
            rng = max(r.max() - r.min(), 1e-6)
            try:
                scores.append(wasserstein_distance(r / rng, s / rng))
            except Exception:
                pass
        return float(np.mean(scores)) if scores else float('inf')

'''
    src = src[:start] + NEW_METHOD + src[end:]
    print("✅ Replaced _wasserstein_numeric_only with normalised version")

# Case 2: uses calculate_average_wasserstein_distance from utils.metrics
elif "calculate_average_wasserstein_distance" in src:
    # Replace the call with an inline normalised version
    OLD_CALL = "            score = calculate_average_wasserstein_distance(real_data, synthetic_data)"
    NEW_CALL = '''            # PATCHED_SC_WASS — inline normalised wasserstein, no external dep
            def _norm_wass(rdf, sdf):
                from scipy.stats import wasserstein_distance as _wd
                ID_KW = ("_id","id_","patient_id","record_id","index","uuid","key","rownum","row_num","seq")
                def _skip(c, v):
                    lo = c.lower()
                    if lo == "id" or any(lo==k or lo.endswith(k) or lo.startswith(k) for k in ID_KW): return True
                    if len(v)>20 and len(np.unique(v))/len(v)>0.95: return True
                    return False
                scores = []
                for col in rdf.select_dtypes(include=[np.number]).columns:
                    if col not in sdf.columns: continue
                    r,s = rdf[col].dropna().values, sdf[col].dropna().values
                    if len(r)<2 or len(s)<2 or _skip(col,r): continue
                    rng = max(r.max()-r.min(), 1e-6)
                    try: scores.append(_wd(r/rng, s/rng))
                    except: pass
                return float(np.mean(scores)) if scores else float('inf')
            score = _norm_wass(real_data, synthetic_data)'''

    if OLD_CALL in src:
        src = src.replace(OLD_CALL, NEW_CALL)
        print("✅ Replaced calculate_average_wasserstein_distance call with inline normalised version")
    else:
        print("⚠️  Could not find exact call — trying regex")
        src = re.sub(
            r'score\s*=\s*calculate_average_wasserstein_distance\(.*?\)',
            NEW_CALL.strip(),
            src
        )
        if "PATCHED_SC_WASS" in src:
            print("✅ Replaced via regex")
        else:
            sys.exit("❌ Could not patch — please paste the wasserstein call line")

try:
    ast.parse(src)
    print("✅ Syntax OK")
except SyntaxError as e:
    sys.exit(f"❌ Syntax error: {e}")

with open(target, "w") as f:
    f.write(src)
print(f"✅ Written: {target}")
print("\nRun: streamlit run app.py")
print("Expected scores: 0.03–0.15 (not 2059)")
