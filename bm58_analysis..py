import argparse
import heapq
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from scipy.ndimage import gaussian_filter, uniform_filter

SEED = 42
N = 36                       
AREA_KM2 = 326.2             
CELL_KM = np.sqrt(AREA_KM2) / N          
CELL_AREA = AREA_KM2 / N**2              
YEARS = np.arange(2000, 2009)
N_WELLS = 12
SY_BASE, SY_LOW, SY_HIGH = 0.02, 0.015, 0.03
WEIGHTS = dict(S=0.25, So=0.25, L=0.20, D=0.15, P=0.15)
CLASS_EDGES = [2.4, 2.9, 3.4]            
CLASS_NAMES = ["Low", "Moderate", "High", "Very high"]
EFFICIENCY = dict(check_dam=0.06, recharge_structure=0.08,
                  farm_pond=0.04, contour_bund=0.05)   
STREAM_THRESHOLD = 25                    
NEIGH = [(-1, -1), (-1, 0), (-1, 1), (0, -1), (0, 1), (1, -1), (1, 0), (1, 1)]

def smooth_noise(rng, sigma):
    f = gaussian_filter(rng.standard_normal((N, N)), sigma, mode="reflect")
    return (f - f.mean()) / f.std()


def make_dem(rng):
    yy, xx = np.mgrid[0:N, 0:N] / (N - 1)        
    tilt = 220 * (1 - xx) + 80 * (1 - yy)
    return 600 + tilt + 85 * smooth_noise(rng, 4) + 30 * smooth_noise(rng, 1.5)


def drainage(dem):
    edge = [(i, j) for i in range(N) for j in range(N) if i in (0, N - 1) or j in (0, N - 1)]
    outlet = min(edge, key=lambda c: dem[c])
    filled = dem.copy()
    recv = -np.ones((N, N, 2), dtype=int)
    seen = np.zeros((N, N), bool)
    seen[outlet] = True
    heap = [(dem[outlet], outlet)]
    order = []                                   
    while heap:
        z, (i, j) = heapq.heappop(heap)
        order.append((i, j))
        for di, dj in NEIGH:
            a, b = i + di, j + dj
            if 0 <= a < N and 0 <= b < N and not seen[a, b]:
                seen[a, b] = True
                filled[a, b] = max(dem[a, b], z + 1e-3)
                recv[a, b] = (i, j)
                heapq.heappush(heap, (filled[a, b], (a, b)))
    acc = np.ones((N, N))
    for (i, j) in reversed(order):               
        r = tuple(recv[i, j])
        if r[0] >= 0:
            acc[r] += acc[i, j]
    stream = acc >= STREAM_THRESHOLD
    strahler = np.zeros((N, N), int)
    donors = {}
    for (i, j) in order:
        r = tuple(recv[i, j])
        if r[0] >= 0 and stream[i, j] and stream[r]:
            donors.setdefault(r, []).append((i, j))
    for (i, j) in reversed(order):
        if not stream[i, j]:
            continue
        ords = [strahler[d] for d in donors.get((i, j), [])]
        if not ords:
            strahler[i, j] = 1
        else:
            m = max(ords)
            strahler[i, j] = m + 1 if ords.count(m) >= 2 else m
    return acc, stream, strahler


def slope_percent(dem):
    gy, gx = np.gradient(dem, CELL_KM * 1000)
    return 100 * np.hypot(gx, gy)


def make_layers(rng):
    dem = make_dem(rng)
    slope = slope_percent(dem)
    acc, stream, strahler = drainage(dem)

    z = smooth_noise(rng, 3.5)
    soil = np.digitize(z, np.quantile(z, [0.15, 0.40, 0.65, 0.85])) + 1
    soil[slope > 12] = np.minimum(soil[slope > 12], 2)        
    soil[stream & (strahler >= 2) & (soil < 4)] = 4           

    z = smooth_noise(rng, 3)
    lulc = np.digitize(z, np.quantile(z, [0.08, 0.28, 0.45, 0.62])) + 1
    lulc[(slope > 10) & (lulc == 5)] = 2

    xx = np.mgrid[0:N, 0:N][1] / (N - 1)
    rain_surf = 640 - 120 * xx + 15 * smooth_noise(rng, 4)

    dd = uniform_filter(stream.astype(float), 5) * CELL_KM / (CELL_KM ** 2)
    return dict(dem=dem, slope=slope, acc=acc, stream=stream, order=strahler,
                soil=soil, lulc=lulc, rain=rain_surf, dd=dd)

def synthetic_wells(rng):
    rain = np.clip(rng.gamma(shape=12, scale=50, size=len(YEARS)), 330, 850)
    well_eff = rng.normal(0, 0.35, N_WELLS)
    rows = []
    for w in range(N_WELLS):
        pre0 = rng.uniform(6, 14)
        for y, r in zip(YEARS, rain):
            dh = max(0.15, 0.0058 * r + well_eff[w] + rng.normal(0, 0.45))
            pre = pre0 + rng.normal(0, 0.5)
            rows.append(dict(well_id=f"W{w+1:02d}", year=int(y), rainfall_mm=round(r, 1),
                             pre_monsoon_m_bgl=round(pre, 2),
                             post_monsoon_m_bgl=round(pre - dh, 2)))
    return pd.DataFrame(rows)


def wtf_recharge(wells):
    wells = wells.copy()
    wells["dh_m"] = wells["pre_monsoon_m_bgl"] - wells["post_monsoon_m_bgl"]   
    wells["dh_m"] = wells["dh_m"].clip(lower=0)
    yr = wells.groupby("year").agg(rainfall_mm=("rainfall_mm", "mean"),
                                   dh_m=("dh_m", "mean")).reset_index()
    a_m2 = AREA_KM2 * 1e6
    for tag, sy in (("base", SY_BASE), ("low", SY_LOW), ("high", SY_HIGH)):
        yr[f"recharge_Mm3_Sy_{tag}"] = sy * yr["dh_m"] * a_m2 / 1e6       
    yr["recharge_mm_base"] = yr["recharge_Mm3_Sy_base"] * 1e6 / a_m2 * 1000
    yr["recharge_pct_rain"] = 100 * yr["recharge_mm_base"] / yr["rainfall_mm"]
    return yr

def scores(L):
    S = np.digitize(L["slope"], [2, 5, 10, 15])                   
    S = 5 - S                                                      
    So = np.choose(L["soil"] - 1, [1, 2, 3, 4, 5]).astype(float)
    Lu = np.choose(L["lulc"] - 1, [1, 2, 4, 3, 5]).astype(float)   
    D = 5 - np.digitize(L["dd"], np.quantile(L["dd"], [0.2, 0.4, 0.6, 0.8]))   
    P = np.digitize(L["rain"], np.quantile(L["rain"], [0.2, 0.4, 0.6, 0.8])) + 1
    return dict(S=S.astype(float), So=So, L=Lu, D=D.astype(float), P=P.astype(float))

def grpi(sc, w=WEIGHTS):
    return sum(w[k] * sc[k] for k in w)


def classify(g):
    return np.digitize(g, CLASS_EDGES)          


def weight_sensitivity(sc, base_cls, rng, runs=200, pert=0.05):
    changed = []
    for _ in range(runs):
        w = {k: v * (1 + rng.uniform(-pert, pert)) for k, v in WEIGHTS.items()}
        s = sum(w.values())
        w = {k: v / s for k, v in w.items()}
        changed.append(100 * np.mean(classify(grpi(sc, w)) != base_cls))
    return float(np.mean(changed)), float(np.max(changed))

def siting(L, cls):
    slope, stream, order = L["slope"], L["stream"], L["order"]
    cropland = L["lulc"] == 5
    near_stream = uniform_filter((stream & (order >= 2)).astype(float), 3) > 0   
    cd = stream & (slope < 8) & (order >= 2)
    rs = (~cd) & near_stream & (cls >= 2) & (slope < 6)
    fp = (~cd) & (~rs) & cropland & (slope < 4) & (L["soil"] <= 3)
    cb = (~cd) & (~rs) & (~fp) & cropland & (slope >= 2) & (slope <= 10)
    out = np.zeros((N, N), int)                     
    out[cb], out[fp], out[rs], out[cd] = 4, 3, 2, 1
    return out


def intervention_table(L, site):
    names = {1: "Check dam", 2: "Recharge structure", 3: "Farm pond", 4: "Contour bunding"}
    keys = {1: "check_dam", 2: "recharge_structure", 3: "farm_pond", 4: "contour_bund"}
    rows = []
    for k in (1, 2, 3, 4):
        m = site == k
        a = m.sum() * CELL_AREA
        p = L["rain"][m].mean() / 1000 if m.any() else 0          
        c = EFFICIENCY[keys[k]]
        gain = a * 1e6 * p * c / 1e6                               
        rows.append(dict(intervention=names[k], treated_area_km2=a,
                         pct_of_watershed=100 * a / AREA_KM2, mean_rainfall_mm=p * 1000,
                         efficiency=c, gain_Mm3_yr=gain,
                         gain_low_Mm3_yr=0.5 * gain, gain_high_Mm3_yr=1.5 * gain,
                         unit_gain_Mm3_per_km2=gain / a if a else 0))
    return pd.DataFrame(rows)


def fig1(L, cls, site, path, shares, show=True):
    ext = [0, N * CELL_KM, 0, N * CELL_KM]
    fig, ax = plt.subplots(1, 2, figsize=(13, 6.3))
    cm = ListedColormap(["#d7191c", "#fdae61", "#a6d96a", "#1a9641"])
    ax[0].imshow(np.flipud(cls), cmap=cm, vmin=0, vmax=3, extent=ext, interpolation="nearest")
    ys, xs = np.where(L["stream"])
    ax[0].scatter((xs + .5) * CELL_KM, (N - ys - .5) * CELL_KM, s=6, c="#08306b", marker="s")
    ax[0].set_title("(a) Groundwater recharge potential (GRPI)", fontweight="bold")
    cm2 = ListedColormap(["#f0f0f0", "#08519c", "#6a3d9a", "#ff7f00", "#33a02c"])
    ax[1].imshow(np.flipud(site), cmap=cm2, vmin=0, vmax=4, extent=ext, interpolation="nearest")
    ax[1].set_title("(b) Suggested intervention siting", fontweight="bold")
    for a in ax:
        a.set_xlabel("Easting within watershed (km)", fontsize=12)
        a.set_ylabel("Northing within watershed (km)", fontsize=12)
        a.tick_params(labelsize=11)
    h1 = [Patch(color=c, label=f"{n} ({s:.0f}%)") for c, n, s in
          zip(["#d7191c", "#fdae61", "#a6d96a", "#1a9641"], CLASS_NAMES, shares)]
    h1.append(Line2D([], [], marker="s", ls="", color="#08306b", label="Main streams"))
    h2 = [Patch(color=c, label=n) for c, n in zip(
        ["#f0f0f0", "#08519c", "#6a3d9a", "#ff7f00", "#33a02c"],
        ["No intervention", "Check dam", "Recharge structure", "Farm pond", "Contour bunding"])]
    fig.legend(handles=h1 + h2, loc="lower center", ncol=5, frameon=False, fontsize=11)
    fig.suptitle("Figure 1. BM-58 watershed: recharge potential and intervention siting (synthetic data)")
    fig.tight_layout(rect=[0, 0.1, 1, 0.96])
    fig.savefig(path, dpi=200)
    if not show:          
        plt.close(fig)


def fig2(yr, itab, path, show=True):
    fig, ax = plt.subplots(1, 2, figsize=(13, 5.6))
    a = ax[0]
    a.bar(yr["year"], yr["recharge_Mm3_Sy_base"], color="#2c7fb8", label="WTF recharge (Sy = 0.02)")
    a.fill_between(yr["year"], yr["recharge_Mm3_Sy_low"], yr["recharge_Mm3_Sy_high"],
                   color="#9ecae1", alpha=.45, label="Sy range 0.015-0.03")
    a.set_xlabel("Year", fontsize=12)
    a.set_ylabel("Annual recharge (million m³)", fontsize=12)
    a.set_title("(a) Baseline WTF recharge and rainfall", fontweight="bold")
    b = a.twinx()
    b.plot(yr["year"], yr["rainfall_mm"], "o-", color="#d95f0e", label="Mean rainfall")
    b.set_ylabel("Rainfall (mm)", fontsize=12)
    h1, l1 = a.get_legend_handles_labels()
    h2, l2 = b.get_legend_handles_labels()
    a.legend(h1 + h2, l1 + l2, loc="upper left", fontsize=10, frameon=False)
    a.set_xticks(yr["year"])
    c = ax[1]
    x = np.arange(len(itab))
    err = [itab["gain_Mm3_yr"] - itab["gain_low_Mm3_yr"], itab["gain_high_Mm3_yr"] - itab["gain_Mm3_yr"]]
    c.bar(x, itab["gain_Mm3_yr"], yerr=err, capsize=5,
          color=["#08519c", "#6a3d9a", "#ff7f00", "#33a02c"])
    c.set_xticks(x)
    c.set_xticklabels(["Check\ndam", "Recharge\nstructure", "Farm\npond", "Contour\nbunding"], fontsize=11)
    c.set_ylabel("Additional recharge (million m³/yr)", fontsize=12)
    c.set_title("(b) Gain per intervention (bars: ±50% efficiency)", fontweight="bold")
    fig.suptitle("Figure 2. Baseline recharge (2000-2008) and intervention scenarios (synthetic data)")
    fig.tight_layout(rect=[0, 0, 1, 0.95])
    fig.savefig(path, dpi=200)
    if not show:          
        plt.close(fig)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--wells", help="CSV: well_id,year,rainfall_mm,pre_monsoon_m_bgl,post_monsoon_m_bgl")
    ap.add_argument("--out", default="outputs")
    ap.add_argument("--no-show", action="store_true",
                    help="only save the figures; do not display them in the editor/window")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)
    rng = np.random.default_rng(SEED)

    layers = make_layers(rng)
    wells = pd.read_csv(args.wells) if args.wells else synthetic_wells(rng)
    wells.to_csv(os.path.join(args.out, "synthetic_well_data_2000_2008.csv" if not args.wells
                              else "wells_used.csv"), index=False)
    yr = wtf_recharge(wells)
    yr.round(3).to_csv(os.path.join(args.out, "annual_recharge_wtf.csv"), index=False)

    sc = scores(layers)
    g = grpi(sc)
    cls = classify(g)
    shares = [100 * np.mean(cls == k) for k in range(4)]
    areas = [s / 100 * AREA_KM2 for s in shares]
    sens_mean, sens_max = weight_sensitivity(sc, cls, np.random.default_rng(SEED + 1))

    site = siting(layers, cls)
    itab = intervention_table(layers, site)
    itab.round(4).to_csv(os.path.join(args.out, "intervention_summary.csv"), index=False)

    cls_tab = pd.DataFrame(dict(potential_class=CLASS_NAMES, area_km2=areas, percent=shares))
    cls_tab.round(2).to_csv(os.path.join(args.out, "potential_class_areas.csv"), index=False)

    fig1(layers, cls, site, os.path.join(args.out, "Figure_1_recharge_potential_and_intervention_map.png"), shares,
         show=not args.no_show)
    fig2(yr, itab, os.path.join(args.out, "Figure_2_baseline_recharge_and_scenarios.png"),
         show=not args.no_show)

    pk, lo = yr.loc[yr["recharge_Mm3_Sy_base"].idxmax()], yr.loc[yr["recharge_Mm3_Sy_base"].idxmin()]
    summary = dict(
        seed=SEED, cell_km=round(CELL_KM, 3), mean_rain_mm=yr["rainfall_mm"].mean(),
        mean_dh_m=yr["dh_m"].mean(), mean_recharge_mm=yr["recharge_mm_base"].mean(),
        mean_recharge_Mm3=yr["recharge_Mm3_Sy_base"].mean(),
        mean_recharge_pct_rain=100 * yr["recharge_mm_base"].mean() / yr["rainfall_mm"].mean(),
        sy_range_Mm3=[yr["recharge_Mm3_Sy_low"].mean(), yr["recharge_Mm3_Sy_high"].mean()],
        peak=dict(year=int(pk.year), Mm3=pk.recharge_Mm3_Sy_base, rain=pk.rainfall_mm),
        minimum=dict(year=int(lo.year), Mm3=lo.recharge_Mm3_Sy_base, rain=lo.rainfall_mm,
                     pct=lo.recharge_pct_rain),
        rain_recharge_corr=float(np.corrcoef(yr["rainfall_mm"], yr["recharge_Mm3_Sy_base"])[0, 1]),
        class_pct=dict(zip(CLASS_NAMES, shares)), class_km2=dict(zip(CLASS_NAMES, areas)),
        untreated_pct=100 * np.mean(site == 0),
        total_gain_Mm3=itab["gain_Mm3_yr"].sum(),
        total_gain_range=[itab["gain_low_Mm3_yr"].sum(), itab["gain_high_Mm3_yr"].sum()],
        gain_pct_of_baseline=100 * itab["gain_Mm3_yr"].sum() / yr["recharge_Mm3_Sy_base"].mean(),
        weight_sensitivity_mean_pct_cells_changed=sens_mean,
        weight_sensitivity_max_pct_cells_changed=sens_max,
        stream_cells=int(layers["stream"].sum()),
    )
    with open(os.path.join(args.out, "results_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=float)
    print(json.dumps(summary, indent=2, default=float))
    print(itab.round(3).to_string(index=False))

    if not args.no_show:
        plt.show()       


if __name__ == "__main__":
    main()
