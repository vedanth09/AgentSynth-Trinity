from typing import Dict, Any, Optional
import pandas as pd
import time
import numpy as np

# SDV Imports
from sdv.single_table import TVAESynthesizer, GaussianCopulaSynthesizer, CTGANSynthesizer
from sdv.metadata import SingleTableMetadata

# Internal Imports
from utils.metrics import calculate_average_wasserstein_distance

# Upgraded Model Imports
from models.timegan_wrapper import ConditionalTimeGANWrapper
from models.ddpm_tabular import DDPMTabular


class ModelLibrary:
    """
    Library of generative models managed by the Statistical Critic.

    Model Overview:
    ---------------
    CTGAN  (GAN)         - Conditional Tabular GAN (Xu et al., 2019). Uses a generator
                           and discriminator trained adversarially. The generator learns
                           to produce synthetic rows that fool the discriminator into
                           thinking they are real. Mode-specific normalisation handles
                           the multi-modal distributions common in tabular data.
                           → Best for: imbalanced columns, rare-event data (fraud, rare diseases).

    TVAE   (VAE)         - Tabular Variational Autoencoder. Encodes real rows into a
                           compressed latent space and samples from it to generate new rows.
                           More training-stable than GANs (no adversarial instability).
                           → Best for: mixed-type tabular data (numeric + categorical).

    GaussianCopula       - Statistical model. Learns column marginals and their correlation
                           structure separately, then samples jointly. Very fast and
                           interpretable — good baseline.
                           → Best for: low-dimensional, roughly Gaussian data.

    DDPM   (Diffusion)   - Denoising Diffusion Probabilistic Model. Iteratively denoises
                           Gaussian noise into realistic synthetic records (Ho et al., 2020).
                           → Architecture implemented; full training loop is future work.
                           → Excluded from live pilot to avoid trivial random-noise scores.

    TimeGAN (GAN)        - Temporal GAN conditioned on market regimes (Yoon et al., 2019).
                           → Architecture implemented; full training loop is future work.
                           → Excluded from live pilot to avoid trivial random-noise scores.
    """

    # Minimum rows needed to train each SDV model reliably
    MIN_ROWS = {"CTGAN": 50, "TVAE": 20, "GaussianCopula": 10}

    def __init__(self):
        self.sdv_models = {
            "CTGAN":          CTGANSynthesizer,
            "TVAE":           TVAESynthesizer,
            "GaussianCopula": GaussianCopulaSynthesizer,
        }
        self.custom_models = {
            "TimeGAN": ConditionalTimeGANWrapper,
            "DDPM":    DDPMTabular,
        }

    # PATCHED_SC_WASS
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


    def train_and_evaluate(self,
                           model_name: str,
                           real_data: pd.DataFrame,
                           metadata: SingleTableMetadata) -> tuple:
        """
        Trains a synthesizer, generates a pilot sample, and returns
        (wasserstein_score, synthetic_dataframe).  Lower score = better fidelity.
        """
        try:
            print(f"   > Piloting {model_name}...")

            # ── Custom model wrappers (DDPM, TimeGAN) ──────────────────────────
            if model_name in self.custom_models:
                SynthClass = self.custom_models[model_name]
                if model_name == "DDPM":
                    synthesizer = SynthClass(input_dim=real_data.shape[1])
                else:
                    synthesizer = SynthClass()
                synthesizer.fit(real_data)
                synthetic_data = synthesizer.sample(num_samples=len(real_data))

            # ── SDV synthesizers (CTGAN, TVAE, GaussianCopula) ─────────────────
            elif model_name in self.sdv_models:
                min_rows = self.MIN_ROWS.get(model_name, 10)
                if len(real_data) < min_rows:
                    print(f"     -> {model_name} skipped: needs ≥{min_rows} rows "
                          f"(got {len(real_data)})")
                    return float('inf'), None

                SynthClass = self.sdv_models[model_name]

                # CTGAN-specific: use more epochs for better GAN convergence
                if model_name == "CTGAN":
                    synthesizer = SynthClass(metadata, epochs=300, verbose=False)
                    print(f"     -> [CTGAN] Training GAN adversarially "
                          f"({len(real_data)} rows, 300 epochs)...")
                else:
                    synthesizer = SynthClass(metadata)

                synthesizer.fit(real_data)
                synthetic_data = synthesizer.sample(num_rows=len(real_data))

            else:
                print(f"     -> Unknown model: {model_name}")
                return float('inf'), None

            # Use numeric-only normalised Wasserstein (avoids million-scale scores)
            score = self._wasserstein_numeric_only(real_data, synthetic_data)
            print(f"     -> {model_name} Wasserstein Score: {score:.4f}")
            return score, synthetic_data

        except Exception as e:
            print(f"     -> {model_name} Failed: {e}")
            return float('inf'), None


class StatisticalCritic:
    """
    Agent responsible for critiquing logic and selecting optimal generative models
    via an 'Experimental Pilot'. Supports Wasserstein-weighted MODEL ENSEMBLING.

    Academic Motivation: Model ensembling achieves better fidelity-utility
    trade-offs by combining diverse generative capabilities (Zhao et al., 2018).
    CTGAN is the primary GAN-based synthesizer (Xu et al., 2019).
    TVAE and GaussianCopula provide complementary generative diversity.
    DDPM and TimeGAN architectures are implemented as future-work extensions.
    """

    def __init__(self):
        self.library = ModelLibrary()

    async def process(self, state: Dict[str, Any]) -> Dict[str, Any]:
        print("[StatisticalCritic] Running Experimental Pilot with Ensembling...")

        raw_data: Optional[pd.DataFrame] = state.get("raw_data")
        domain     = state.get("domain", "General")
        user_model = state.get("selected_model_type", "auto")

        if raw_data is None:
            return state

        # Rebuild metadata fresh each call (avoids stale schema across iterations)
        metadata = SingleTableMetadata()
        metadata.detect_from_dataframe(raw_data)

        # ── 1. Select candidate models ──────────────────────────────────────────
        # CTGAN (GAN) is always included — it is the primary GAN contribution.
        # DDPM and TimeGAN are architecture stubs with random-noise output — they
        # are excluded from the live pilot because their trivial Wasserstein=0
        # would always win the competition, producing meaningless synthetic data.
        # They are documented as future-work extensions in the thesis.
        if user_model != "auto":
            model_map = {
                "timegan":   ["CTGAN", "TVAE"],       # TimeGAN stub → fallback to CTGAN
                "vae":       ["TVAE", "GaussianCopula"],
                "diffusion": ["CTGAN", "TVAE"],       # DDPM stub → fallback to CTGAN
            }
            candidates = model_map.get(user_model, ["CTGAN", "TVAE", "GaussianCopula"])
        else:
            # Auto mode: always run the 3 real models
            candidates = ["CTGAN", "TVAE", "GaussianCopula"]

        print(f"   > Candidate models for {domain}: {candidates}")

        # ── 2. Run pilot on a sample (cap at 500 rows for speed) ───────────────
        data_sample = raw_data.sample(min(len(raw_data), 500), random_state=42)

        pilot_results   = {}   # model_name → wasserstein score
        samples_library = {}   # model_name → synthetic DataFrame

        for model_name in candidates:
            score, samples = self.library.train_and_evaluate(
                model_name, data_sample, metadata
            )
            if samples is not None:
                pilot_results[model_name]   = score
                samples_library[model_name] = samples

        if not pilot_results:
            print("[StatisticalCritic] All models failed. Returning raw data as fallback.")
            state["model_selection"] = "Fallback"
            state["safe_data_asset"] = raw_data
            state["pilot_metrics"]   = {}
            return state

        # ── 3. Wasserstein-weighted ensembling of top-2 models ─────────────────
        sorted_models = sorted(pilot_results.items(), key=lambda x: x[1])

        if len(sorted_models) >= 2:
            m1, s1 = sorted_models[0]
            m2, s2 = sorted_models[1]

            # Inverse-Wasserstein weighting: lower score → higher weight
            w1 = 1.0 / (s1 + 1e-6)
            w2 = 1.0 / (s2 + 1e-6)
            norm_w1 = w1 / (w1 + w2)

            print(f"[StatisticalCritic] Ensembling {m1} ({norm_w1:.1%}) "
                  f"+ {m2} ({1 - norm_w1:.1%})")

            n1 = int(len(raw_data) * norm_w1)
            n2 = len(raw_data) - n1

            s1_df = samples_library[m1]
            s2_df = samples_library[m2]
            n1 = min(n1, len(s1_df))
            n2 = min(n2, len(s2_df))

            ensemble_sample = pd.concat([
                s1_df.sample(n1, replace=len(s1_df) < n1, random_state=42),
                s2_df.sample(n2, replace=len(s2_df) < n2, random_state=42),
            ]).reset_index(drop=True)

            state["model_selection"] = f"Ensemble({m1}+{m2})"
            state["safe_data_asset"] = ensemble_sample

        else:
            best_model = sorted_models[0][0]
            print(f"[StatisticalCritic] Single best model: {best_model}")
            state["model_selection"] = best_model
            state["safe_data_asset"] = samples_library[best_model]

        state["pilot_metrics"] = pilot_results

        # ── Store pilot_scores for the Streamlit showcase tab ──────────────────
        # Each entry: {model, wasserstein, tstr (None — critic doesn't compute), mia (None)}
        state["pilot_scores"] = [
            {"model": name, "wasserstein": score, "tstr": None, "mia": None}
            for name, score in sorted_models
        ]

        # Ensemble weights as {model_name: weight} for the leaderboard
        if len(sorted_models) >= 2:
            m1, s1 = sorted_models[0]
            m2, s2 = sorted_models[1]
            w1 = 1.0 / (s1 + 1e-6)
            w2 = 1.0 / (s2 + 1e-6)
            nw1 = w1 / (w1 + w2)
            state["ensemble_weights"] = {m1: nw1, m2: 1 - nw1}
        else:
            state["ensemble_weights"] = {sorted_models[0][0]: 1.0} if sorted_models else {}

        # ── agent_trace entry for the Gantt chart ─────────────────────────────
        existing_trace = state.get("agent_trace") or []
        existing_trace.append({
            "agent":      "StatisticalCritic",
            "output":     state.get("model_selection", ""),
            "duration_s": round(len(candidates) * 3.5, 1),  # estimated
        })
        state["agent_trace"] = existing_trace

        # Print leaderboard
        print("\n   ── Model Pilot Leaderboard (lower Wasserstein = better) ──")
        for rank, (name, score) in enumerate(sorted_models, 1):
            tag = " ← GAN" if name == "CTGAN" else \
                  " ← VAE" if name == "TVAE" else \
                  " ← Diffusion" if name == "DDPM" else \
                  " ← Temporal GAN" if name == "TimeGAN" else ""
            print(f"   #{rank}  {name:<18} Wasserstein={score:.4f}{tag}")
        print()

        return state