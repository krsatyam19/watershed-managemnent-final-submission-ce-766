# BM-58 Watershed: Groundwater Recharge Assessment and Intervention Siting (2000-2008)

Final project, BM-58 watershed (Purandar and Baramati talukas, Pune, Maharashtra).
The project estimates baseline groundwater recharge, maps recharge potential and sites
watershed interventions (check dams, recharge structures, farm ponds, contour bunding).

**Data note:** observed GSDA / IMD / Bhuvan / ISRIC data could not be processed in time, so
the analysis runs on a **synthetic dataset** (random seed 42) mimicking semi-arid Deccan
basalt conditions. Results are illustrative, not observed values.

## Folder structure

```
bm58-groundwater-recharge/
├── README.md
├── requirements.txt
├── .gitignore
├── src/
│   └── bm58_analysis.py          # complete analysis code
├── data/
│   └── synthetic_well_data_2000_2008.csv   # synthetic rainfall and well levels (12 wells)
├── results/
│   ├── results_summary.json
│   ├── figures/
│   │   ├── Figure_1_recharge_potential_and_intervention_map.png
│   │   └── Figure_2_baseline_recharge_and_scenarios.png
│   └── tables/
│       ├── annual_recharge_wtf.csv
│       ├── potential_class_areas.csv
│       └── intervention_summary.csv
└── docs/
    └── BM-58_Final_Project_Submission.docx
```

## Methods (see docs/ for full description)

1. Water Table Fluctuation recharge: R = Sy * dh * A (Healy and Cook, 2002)
2. Groundwater Recharge Potential Index: GRPI = 0.25*S + 0.25*So + 0.20*L + 0.15*D + 0.15*P (after Shaban et al., 2006)
3. Rule-based intervention siting with recharge gain dR_k = A_k * P_k * c_k

## How to run

```bash
pip install -r requirements.txt
python src/bm58_analysis.py              # saves outputs and shows the plots
python src/bm58_analysis.py --no-show    # saves outputs only
```

Running regenerates everything in `data/` and `results/` (identical every time because of the fixed seed).

To use observed records instead of synthetic data:

```bash
python src/bm58_analysis.py --wells data/wells_observed.csv
```

with columns `well_id,year,rainfall_mm,pre_monsoon_m_bgl,post_monsoon_m_bgl`.

## References

- Healy, R.W. and Cook, P.G. (2002). Using groundwater levels to estimate recharge. Hydrogeology Journal, 10(1), 91-109. https://doi.org/10.1007/s10040-001-0178-0
- Shaban, A., Khawlie, M. and Abdallah, C. (2006). Use of remote sensing and GIS to determine recharge potential zones: the case of Occidental Lebanon. Hydrogeology Journal, 14, 433-443. https://doi.org/10.1007/s10040-005-0437-6
- Central Ground Water Board (CGWB). Report of the Ground Water Resource Estimation Committee (GEC-2015). https://cgwb.gov.in/
