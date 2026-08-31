"""
Builds validation_data/*.csv and *_params.csv from raw values transcribed
directly from the three source papers. Run once; output is checked into
the repo (not regenerated at import time) so reviewers can diff the CSVs
against the cited tables by hand.

Sources:
  Paper 1: Wisniowski, Skrzypaszek & Malachowski, Energies 2020, 13, 3192.
           DOI: 10.3390/en13123192  (Tables 1-6)
  Paper 2: Wisniowski, Skrzypaszek & Toczek, Energies 2022, 15, 5583.
           DOI: 10.3390/en15155583  (Table 7, Table 8)
  Paper 3: Anawe & Folayan, Data in Brief 21 (2018) 289-298.
           DOI: 10.1016/j.dib.2018.09.100  (Tables 1, 3, 4, 6)
"""
import csv
import os

OUT_DIR = os.path.join(os.path.dirname(__file__), "..", "validation_data")
os.makedirs(OUT_DIR, exist_ok=True)

SHEAR_RATE_FACTOR = 1.703   # rpm -> 1/s, per project convention
STRESS_FACTOR = 0.511       # dial reading (deg) -> Pa, per project convention


def write_raw_csv(filename, source_note, rpm_list, dial_list):
    assert len(rpm_list) == len(dial_list)
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", newline="") as f:
        f.write(f"# {source_note}\n")
        w = csv.writer(f)
        w.writerow(["rpm", "dial_reading", "shear_rate_1s",
                     "shear_stress_pa", "shear_stress_lbf100ft2"])
        for rpm, dial in zip(rpm_list, dial_list):
            shear_rate = round(rpm * SHEAR_RATE_FACTOR, 4)
            shear_stress_pa = round(dial * STRESS_FACTOR, 4)
            w.writerow([rpm, dial, shear_rate, shear_stress_pa, dial])
    print("wrote", path)


def write_params_csv(filename, source_note, rows):
    """rows: list of dicts with keys model, param_name, param_value, unit, r2"""
    path = os.path.join(OUT_DIR, filename)
    with open(path, "w", newline="") as f:
        f.write(f"# {source_note}\n")
        w = csv.DictWriter(f, fieldnames=["model", "param_name", "param_value", "unit", "r2"])
        w.writeheader()
        for row in rows:
            w.writerow(row)
    print("wrote", path)


# ---------------------------------------------------------------------------
# PAPER 1 (Energies 2020, 13, 3192) -- Table 1/2 raw, Table 3/4 params, Table 5/6 R
# ---------------------------------------------------------------------------
P1_SRC = "Wisniowski et al., Energies 2020, 13, 3192, DOI: 10.3390/en13123192"
RPM12 = [1, 2, 3, 6, 10, 20, 30, 60, 100, 200, 300, 600]

# Raw dial readings (degrees), transcribed from Table 1 / cross-checked against
# Table 2 (shear_stress_pa = dial * 0.511)
p1_dial = {
    "cement_wc050_plain":     [12, 15, 18, 25, 32, 47, 54, 67, 80, 108, 132, 183],
    "cement_wc050_psp042":    [2, 2, 3, 4, 6, 10, 14, 28, 48, 91, 141, 237],
    "bentonite3pct_plain":    [3, 4, 5, 6, 6, 6, 6, 8, 10, 14, 18, 28],
    "bentonite3pct_xcd2pct":  [23, 24, 26, 28, 31, 35, 38, 45, 51, 65, 76, 98],
}

for name, dials in p1_dial.items():
    write_raw_csv(f"paper1_{name}.csv", f"{P1_SRC}, Table 1/2 -- fluid: {name}", RPM12, dials)

# Published fitted params, Table 3 (cement pair) & Table 4 (mud pair);
# R values (correlation coeff) from Table 5 (mud pair) & Table 6 (cement pair) -> r2 = R^2
p1_params = {
    "cement_wc050_plain": {
        "Newtonian":       {"eta": (0.1105, "Pa.s")},
        "Bingham":         {"PV": (0.0842, "Pa.s"), "YP": (16.9723, "Pa")},
        "PowerLaw":        {"K": (4.7932, "Pa.s^n"), "n": (0.4265, "-")},
        "Casson":          {"eta_cas": (0.0532, "Pa.s"), "tau0": (8.7314, "Pa")},
        "HerschelBulkley": {"K": (2.2527, "Pa.s^n"), "n": (0.5420, "-"), "tau0": (1.2854, "Pa")},
        "VomBerg":         {"tau0": (9.8527, "Pa"), "D": (26.5833, "Pa"), "C": (105.2867, "1/s")},
        "HahnEyring":      {"E": (0.0922, "Pa.s"), "D": (1.6225, "Pa"), "C": (0.7939, "1/s")},
        "r2": {
            "Newtonian": 0.77729 ** 2, "Bingham": 0.95144 ** 2, "PowerLaw": 0.99855 ** 2,
            "Casson": 0.97705 ** 2, "HerschelBulkley": 0.98691 ** 2,
            "VomBerg": 0.99009 ** 2, "HahnEyring": 0.90418 ** 2,
        },
    },
    "cement_wc050_psp042": {
        "Newtonian":       {"eta": (0.1246, "Pa.s")},
        "Bingham":         {"PV": (0.1215, "Pa.s"), "YP": (1.9507, "Pa")},
        "PowerLaw":        {"K": (0.3931, "Pa.s^n"), "n": (0.8012, "-")},
        "Casson":          {"eta_cas": (0.1159, "Pa.s"), "tau0": (0.1668, "Pa")},
        "HerschelBulkley": {"K": (0.2925, "Pa.s^n"), "n": (0.8718, "-"), "tau0": (0.1440, "Pa")},
        "VomBerg":         {"tau0": (0.0165, "Pa"), "D": (96.9329, "Pa"), "C": (643.4336, "1/s")},
        "HahnEyring":      {"E": (0.1146, "Pa.s"), "D": (0.9401, "Pa"), "C": (1.0184, "1/s")},
        "r2": {
            "Newtonian": 0.99534 ** 2, "Bingham": 0.99639 ** 2, "PowerLaw": 0.98011 ** 2,
            "Casson": 0.99716 ** 2, "HerschelBulkley": 0.99874 ** 2,
            "VomBerg": 0.99956 ** 2, "HahnEyring": 0.99692 ** 2,
        },
    },
    "bentonite3pct_plain": {
        "Newtonian":       {"eta": (0.0159, "Pa.s")},
        "Bingham":         {"PV": (0.0119, "Pa.s"), "YP": (2.5903, "Pa")},
        "PowerLaw":        {"K": (1.3046, "Pa.s^n"), "n": (0.2959, "-")},
        "Casson":          {"eta_cas": (0.0057, "Pa.s"), "tau0": (1.7405, "Pa")},
        "HerschelBulkley": {"K": (0.1682, "Pa.s^n"), "n": (0.6217, "-"), "tau0": (1.3479, "Pa")},
        "VomBerg":         {"tau0": (0.4649, "Pa"), "D": (1.8184, "Pa"), "C": (9.2868, "1/s")},
        "HahnEyring":      {"E": (0.0086, "Pa.s"), "D": (0.6916, "Pa"), "C": (1.0548, "1/s")},
        "r2": {
            "Newtonian": 0.77745 ** 2, "Bingham": 0.99074 ** 2, "PowerLaw": 0.92387 ** 2,
            "Casson": 0.99554 ** 2, "HerschelBulkley": 0.98992 ** 2,
            "VomBerg": 0.88438 ** 2, "HahnEyring": 0.98781 ** 2,
        },
    },
    "bentonite3pct_xcd2pct": {
        "Newtonian":       {"eta": (0.0622, "Pa.s")},
        "Bingham":         {"PV": (0.0377, "Pa.s"), "YP": (15.8508, "Pa")},
        "PowerLaw":        {"K": (8.9200, "Pa.s^n"), "n": (0.2241, "-")},
        "Casson":          {"eta_cas": (0.0145, "Pa.s"), "tau0": (11.8160, "Pa")},
        "HerschelBulkley": {"K": (4.6678, "Pa.s^n"), "n": (0.3285, "-"), "tau0": (2.8143, "Pa")},
        "VomBerg":         {"tau0": (12.6203, "Pa"), "D": (11.5721, "Pa"), "C": (98.7531, "1/s")},
        "HahnEyring":      {"E": (0.0244, "Pa.s"), "D": (2.4781, "Pa"), "C": (0.0507, "1/s")},
        "r2": {
            "Newtonian": 0.0, "Bingham": 0.96043 ** 2, "PowerLaw": 0.97087 ** 2,
            "Casson": 0.99417 ** 2, "HerschelBulkley": 0.98816 ** 2,
            "VomBerg": 0.99375 ** 2, "HahnEyring": 0.99679 ** 2,
        },
    },
}

for name, models in p1_params.items():
    rows = []
    r2map = models.pop("r2")
    for model, params in models.items():
        for pname, (val, unit) in params.items():
            rows.append({
                "model": model, "param_name": pname, "param_value": val,
                "unit": unit, "r2": round(r2map.get(model, ""), 5) if r2map.get(model, "") != "" else "",
            })
    write_params_csv(f"paper1_{name}_params.csv", f"{P1_SRC}, Table 3/4 (params), Table 5/6 (R->r2) -- fluid: {name}", rows)


# ---------------------------------------------------------------------------
# PAPER 2 (Energies 2022, 15, 5583) -- Table 7 raw (drilling mud), Table 8 params
# NOTE: Table 8 only reports Vom Berg / Hahn-Eyring params for this specific
# dataset (computed at LOW/MID/TOP shear-rate triplets for two flow
# geometries). No Bingham/PowerLaw/HerschelBulkley ground truth exists for
# this dataset in the paper -- do not fabricate it.
# ---------------------------------------------------------------------------
P2_SRC = "Wisniowski, Skrzypaszek & Toczek, Energies 2022, 15, 5583, DOI: 10.3390/en15155583"

p2_rpm =  [0.9, 1.8, 3, 6, 30, 60, 90, 100, 180, 200, 300, 600]
# NOTE: paper's Table 7 lists torsion angle 52 at 180 RPM, but its own
# "Shear stresses" row gives tau=26.06 Pa there; 52*0.511=26.57 (off by
# 0.5 Pa), while 51*0.511=26.061 matches the published tau to 3dp. Every
# other one of the 12 rows reconciles exactly with dial*0.511. Treating
# this as a transcription typo in the source and using 51, so that
# shear_stress_pa in this CSV matches the paper's own published values.
p2_dial = [23, 24, 26, 28, 31, 35, 38, 45, 51, 65, 76, 98]
write_raw_csv("paper2_drilling_mud_fann35a.csv",
              f"{P2_SRC}, Table 7 -- drilling mud, Fann 35A/SR-12, R1-B1, spring F1 "
              f"(dial=51 at 180 RPM corrected from apparent source typo 52)",
              p2_rpm, p2_dial)

# Table 8: params determined at three shear-rate triplets (LOW/MID/TOP) for
# two flow geometries (drill pipe interior, annulus). Recorded here as-is;
# these are NOT generic curve-fit params over the full 12-point dataset, so
# tests against them should use the same LOW/MID/TOP subsetting logic if
# comparing directly, or treat them as a documented reference case only.
p2_rows = [
    # Interior of 3-1/2" drill pipe: gamma_LOW=340.48, gamma_MID=511.02, gamma_TOP=1022.04
    {"model": "VomBerg_pipe",    "param_name": "tau0", "param_value": 17.55255355, "unit": "Pa", "r2": ""},
    {"model": "VomBerg_pipe",    "param_name": "D",    "param_value": 18.10492882, "unit": "Pa", "r2": ""},
    {"model": "VomBerg_pipe",    "param_name": "C",    "param_value": 348.6600,    "unit": "1/s", "r2": ""},
    {"model": "HahnEyring_pipe", "param_name": "E",    "param_value": 0.00732346,  "unit": "Pa.s", "r2": ""},
    {"model": "HahnEyring_pipe", "param_name": "D",    "param_value": 10.83764167, "unit": "Pa", "r2": ""},
    {"model": "HahnEyring_pipe", "param_name": "C",    "param_value": 40.1643,     "unit": "1/s", "r2": ""},
    # Annulus 3-1/2" pipe -- 9-5/8" casing: gamma_LOW=51.10, gamma_MID=102.02, gamma_TOP=153.31
    {"model": "VomBerg_annulus",    "param_name": "tau0", "param_value": 13.25003780, "unit": "Pa", "r2": ""},
    {"model": "VomBerg_annulus",    "param_name": "D",    "param_value": 4.59896869,  "unit": "Pa", "r2": ""},
    {"model": "VomBerg_annulus",    "param_name": "C",    "param_value": 86.080000,   "unit": "1/s", "r2": ""},
    {"model": "HahnEyring_annulus", "param_name": "E",    "param_value": 0.01590531,  "unit": "Pa.s", "r2": ""},
    {"model": "HahnEyring_annulus", "param_name": "D",    "param_value": 1.77624900,  "unit": "Pa", "r2": ""},
    {"model": "HahnEyring_annulus", "param_name": "C",    "param_value": 0.021630,    "unit": "1/s", "r2": ""},
]
write_params_csv("paper2_drilling_mud_fann35a_params.csv",
                  f"{P2_SRC}, Table 8 -- Vom Berg / Hahn-Eyring ONLY (no Bingham/PL/HB ground truth published for this dataset)",
                  p2_rows)


# ---------------------------------------------------------------------------
# PAPER 3 (Anawe & Folayan, Data in Brief 2018) -- Table 1 raw, Tables 2-6 params
# Dial readings given directly in lbf/100ft^2 (== dial_reading by convention
# here; shear_stress_pa computed via *0.511 same as other datasets for
# internal consistency, even though the paper's own model equations were fit
# directly in Pa using a slightly different lb/100ft^2->Pa factor -- see note).
# ---------------------------------------------------------------------------
P3_SRC = "Anawe & Folayan, Data in Brief 21 (2018) 289-298, DOI: 10.1016/j.dib.2018.09.100"
p3_rpm = [600, 300, 200, 100, 60, 30, 6, 3]

p3_dial = {
    "80F":  [82, 55, 43, 34, 25, 20, 15, 11],
    "120F": [61.5, 42.5, 33, 25, 19, 14.5, 10, 7],
    "160F": [44, 31, 24, 18, 13.5, 10, 8, 5],
    "200F": [32, 23, 19, 15, 10.5, 7, 5, 3.5],
}
for temp, dials in p3_dial.items():
    write_raw_csv(f"paper3_bentonite_mud_{temp}.csv",
                   f"{P3_SRC}, Table 1 -- bentonite-gel water-based mud at {temp}", p3_rpm, dials)

# Bingham (Table 3), Power Law (Table 4), Herschel-Bulkley (Table 6)
p3_params = {
    "80F":  {"Bingham": {"YP": 14.308, "PV": 0.0138},
             "PowerLaw": {"n": 0.576, "K": 1.5146},
             "HerschelBulkley": {"tau0": 5.621, "n": 0.6335, "K": 0.4236}},
    "120F": {"Bingham": {"YP": 12.008, "PV": 0.0097},
             "PowerLaw": {"n": 0.533, "K": 1.5304},
             "HerschelBulkley": {"tau0": 3.577, "n": 0.6366, "K": 0.3340}},
    "160F": {"Bingham": {"YP": 9.198, "PV": 0.00664},
             "PowerLaw": {"n": 0.505, "K": 1.329},
             "HerschelBulkley": {"tau0": 2.555, "n": 0.5831, "K": 0.3263}},
    "200F": {"Bingham": {"YP": 7.154, "PV": 0.004599},
             "PowerLaw": {"n": 0.476, "K": 1.182},
             "HerschelBulkley": {"tau0": 1.788, "n": 0.664, "K": 0.1595}},
}
# R^2 values read off Fig. 3 (Newtonian), Fig. 4 (Herschel-Bulkley log-log fit)
# in the paper; Bingham/PowerLaw R^2 not explicitly tabulated by the paper
# for this dataset, left blank rather than fabricated.
p3_hb_r2 = {"80F": 0.9926, "120F": 0.9981, "160F": 0.9771, "200F": 0.9865}

units = {"YP": "Pa", "PV": "Pa.s", "n": "-", "K": "Pa.s^n", "tau0": "Pa"}
for temp, models in p3_params.items():
    rows = []
    for model, params in models.items():
        for pname, val in params.items():
            r2 = p3_hb_r2[temp] if model == "HerschelBulkley" else ""
            rows.append({"model": model, "param_name": pname, "param_value": val,
                         "unit": units[pname], "r2": r2})
    write_params_csv(f"paper3_bentonite_mud_{temp}_params.csv",
                      f"{P3_SRC}, Table 3 (Bingham), Table 4 (PowerLaw), Table 6 (HerschelBulkley) -- {temp}",
                      rows)

print("done")
