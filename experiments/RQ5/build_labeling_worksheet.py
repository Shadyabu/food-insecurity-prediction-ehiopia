"""RQ5 Phase 2 — candidate case list, curated by reading 5 FEWS NET Ethiopia
Food Security Outlook/Update reports spanning the model's actual test
window (2020-01 to 2022-12, the default split RQ5's SHAP cache uses --
NOT the 2022-2026 window in docs/dissertation_plan.md's literal text,
which the model_join pipeline already deviates from for the same
Busker-parity reason; see CLAUDE.md section 2's model_join entry). Reports
read: June 2020, February 2021, October 2021, June 2022, October 2022,
plus August 2020 (Outlook Update) specifically for a flood case, since none
of the other 5 named flooding as a primary "area of concern" driver.

Every "areas of concern" reading below is a genuine, sourced paraphrase of
what each report's WebFetch extraction returned -- not fabricated. The
zone_code / cluster mapping from FEWS NET's region/zone names to this
project's 92 admin2 zone_code key IS my own work (checked against
boundaries/livelihood_zones_admin2.csv), and the candidate_taxonomy_cause
column is MY proposed reading, offered as a starting point -- it is NOT
the ground_truth_cause column, which the project owner fills in
independently after reading the source report directly (Phase 2 step 2,
manual by design -- narrative extraction is not automated in this
project).

This produces the worksheet the project owner reads/edits, then labels
(Phase 3 requires a second, blind pass over the same set).
"""

import csv
from pathlib import Path

RQ5_DIR = Path(__file__).resolve().parent

# Each row: case_id, zone_codes (comma-sep, must exist in ethiopia_lead*.csv),
# region, time_period (within the 2020-01..2022-12 test window), report_url,
# source_quote_or_paraphrase, candidate_taxonomy_cause (my proposed reading,
# not ground truth).
CASES = [
    dict(
        case_id="C01_borena_drought_2021Q4",
        zone_codes="ET0412",
        zone_names="Borena",
        region="Oromia",
        time_period="2021-10 to 2021-12",
        report_url="https://fews.net/east-africa/ethiopia/food-security-outlook/october-2021",
        source_quote='"rainfall was at least 60 percent lower than normal" with "driest conditions for October in forty years"; Crisis (IPC Phase 3), Emergency expected February 2022',
        candidate_taxonomy_cause="DROUGHT",
    ),
    dict(
        case_id="C02_somali_pastoral_drought_2021Q4",
        zone_codes="ET0511,ET0509,ET0508",
        zone_names="Daawa, Liban, Afder",
        region="Somali",
        time_period="2021-10 to 2021-12",
        report_url="https://fews.net/east-africa/ethiopia/food-security-outlook/october-2021",
        source_quote='same drought driver as Borena: "prolonged drought and poor rainfall", Crisis (IPC Phase 3), Emergency expected February 2022',
        candidate_taxonomy_cause="DROUGHT",
    ),
    dict(
        case_id="C03_ssepastoral_drought_2022Q2",
        zone_codes="ET0412,ET0508,ET0511,ET0509,ET0506",
        zone_names="Borena, Afder, Daawa, Liban, Shabelle",
        region="Oromia/Somali",
        time_period="2022-06 to 2022-08",
        report_url="https://fews.net/east-africa/ethiopia/food-security-outlook/june-2022",
        source_quote='"unprecedented fifth consecutive poor rainfall season"; "Emergency (IPC Phase 4) is also expected to be widespread in southern and southeastern pastoral areas"; livestock deaths ~3.4 million as of early June',
        candidate_taxonomy_cause="DROUGHT",
    ),
    dict(
        case_id="C04_ssepastoral_drought_2022Q4",
        zone_codes="ET0412,ET0508,ET0511,ET0509,ET0506,ET0504,ET0503,ET0507,ET0505",
        zone_names="Borena, Dawa, Liban, Afder, Shabelle, Erer, Jarar, Doolo, Korahe",
        region="Oromia/Somali",
        time_period="2022-10 to 2022-12",
        report_url="https://fews.net/east-africa/ethiopia/food-security-outlook/october-2022",
        source_quote='"The current drought, which is unprecedented on the available historical record, is expected to persist through at least mid-2023"; ~4.5 million livestock deaths since October 2021; Emergency Phase 4! / Crisis Phase 3!',
        candidate_taxonomy_cause="DROUGHT",
    ),
    dict(
        case_id="C05_tigray_conflict_2021Q1",
        zone_codes="ET0101,ET0102,ET0103,ET0104,ET0105,ET0106,ET0107",
        zone_names="all 7 Tigray admin2 zones",
        region="Tigray",
        time_period="2021-02 to 2021-04",
        report_url="https://fews.net/east-africa/ethiopia/food-security-outlook/february-2021",
        source_quote='"Large food consumption gaps indicative of Emergency (IPC Phase 4) outcomes...conflict will continue to significantly limit access to typical sources of income" (central/eastern Tigray); Crisis (Phase 3) in southern/central/northwestern Tigray, "conflict is still a driver"',
        candidate_taxonomy_cause="CONFLICT",
    ),
    dict(
        case_id="C06_tigray_conflict_2022Q2",
        zone_codes="ET0101,ET0102,ET0103,ET0104,ET0105,ET0106,ET0107",
        zone_names="all 7 Tigray admin2 zones",
        region="Tigray",
        time_period="2022-06 to 2022-08",
        report_url="https://fews.net/east-africa/ethiopia/food-security-outlook/june-2022",
        source_quote='"Emergency (IPC Phase 4) outcomes, at a minimum, are expected to be widespread" due to the humanitarian truce\'s limited impact; "seasonal labor migration...is not a viable option due to insecurity"',
        candidate_taxonomy_cause="CONFLICT",
    ),
    dict(
        case_id="C07_amhara_border_conflict_2021Q4",
        zone_codes="ET0308,ET0303",
        zone_names="Wag Hamra, North Wello",
        region="Amhara",
        time_period="2021-10 to 2021-12",
        report_url="https://fews.net/east-africa/ethiopia/food-security-outlook/october-2021",
        source_quote='"conflict and continued presence of armed combatants has most significantly disrupted livelihood activities and led to the destruction of assets"; Emergency (IPC Phase 4)',
        candidate_taxonomy_cause="CONFLICT",
    ),
    dict(
        case_id="C08_amhara_border_conflict_2022Q4",
        zone_codes="ET0308,ET0303",
        zone_names="Wag Hamra, North Wello",
        region="Amhara",
        time_period="2022-10 to 2022-12",
        report_url="https://fews.net/east-africa/ethiopia/food-security-outlook/october-2022",
        source_quote='"Conflict resulted in reduced engagement in the main crop production season...households are expected to harvest fewer food stocks than usual and deplete them earlier than normal"; Emergency Phase 4 in worst-affected areas',
        candidate_taxonomy_cause="CONFLICT",
    ),
    dict(
        case_id="C09_afar_zone2_zone4_conflict_2021Q4",
        zone_codes="ET0202,ET0204",
        zone_names="Kilbati-Zone 2, Fanti-Zone 4",
        region="Afar",
        time_period="2021-10 to 2021-12",
        report_url="https://fews.net/east-africa/ethiopia/food-security-outlook/october-2021",
        source_quote='"large-scale livestock losses are substantially limiting households\' capacity to earn income" -- report attributes to conflict-driven displacement + market disruption, not drought; Emergency (IPC Phase 4)',
        candidate_taxonomy_cause="CONFLICT",
    ),
    dict(
        case_id="C10_southern_afar_flooding_2020Q3",
        zone_codes="ET0201,ET0203",
        zone_names="Awsi-Zone 1, Gabi-Zone 3 (woredas Amibara/Asayita/Afambo/Dupty/Mille/Gewane/Awash Fentale span both)",
        region="Afar",
        time_period="2020-08 to 2020-10",
        report_url="https://fews.net/east-africa/ethiopia/food-security-outlook-update/august-2020",
        source_quote='"Crisis (IPC Phase 3) outcomes have emerged in some southern areas of Afar as flooding has led to displacement of people, damage of crop fields, and the loss of livestock"; over 139,000 people displaced by mid-August across Afar/Somali/Oromia/SNNP/Gambela/Amhara. Window widened to 2020-10 (vs. the report'"'"'s own Aug-Sep 2020 language) because IPC assessments for these zones only land in Feb/Jun/Oct -- no observed row exists inside Aug-Sep; October is the nearest real assessment likely to reflect the flooding'"'"'s aftermath.',
        candidate_taxonomy_cause="FLOODING",
    ),
    dict(
        case_id="C11_metekel_conflict_2021Q1",
        zone_codes="ET0602",
        zone_names="Metekel",
        region="Benishangul Gumuz",
        time_period="2021-02 to 2021-04",
        report_url="https://fews.net/east-africa/ethiopia/food-security-outlook/february-2021",
        source_quote='"Crisis (IPC Phase 3) outcomes are ongoing in this area due to conflict and the subsequent disruption to livelihood and economic activities"',
        candidate_taxonomy_cause="CONFLICT",
    ),
    dict(
        case_id="C12_snnp_sidama_hararge_compound_2022Q2",
        zone_codes="ET1600,ET0410,ET0409",
        zone_names="Sidama, East Hararge, West Hararge",
        region="SNNP/Oromia",
        time_period="2022-06 to 2022-08",
        report_url="https://fews.net/east-africa/ethiopia/food-security-outlook/june-2022",
        source_quote='"households deplete below-average harvests earlier than normal and face high market prices" -- report explicitly names BOTH a failed-rains driver and a price/market driver together, a natural COMPOUND candidate; Crisis (IPC Phase 3) until at least October',
        candidate_taxonomy_cause="COMPOUND (DROUGHT + MARKET_SHOCK)",
    ),
    dict(
        case_id="C13_western_oromia_southomo_conflict_2022Q4",
        zone_codes="ET0707,ET0414,ET0415,ET0723,ET0713",
        zone_names="South Omo, Guji, West Guji, Derashe, Konso",
        region="Oromia/SNNP",
        time_period="2022-10 to 2022-12",
        report_url="https://fews.net/east-africa/ethiopia/food-security-outlook/october-2022",
        source_quote='"Conflict in western Oromia; Guji, West Guji of southern Oromia; and South Omo, Derashe, and Konso...was high in October...trade movement and humanitarian access are limited"',
        candidate_taxonomy_cause="CONFLICT",
    ),
    dict(
        case_id="C14_wagHimra_hararghe_covid_market_2020Q2",
        zone_codes="ET0308,ET0410,ET0409",
        zone_names="Wag Hamra, East Hararge, West Hararge",
        region="Amhara/Oromia",
        time_period="2020-06 to 2020-08",
        report_url="https://fews.net/east-africa/ethiopia/food-security-outlook/june-2020",
        source_quote='"people and livestock movement are significantly restricted...households still face food consumption gaps" (Wag Himra: COVID-19 restrictions + drought recovery + high prices + desert locusts); Hararghe lowlands: "low income from casual and agriculture labor given restricted movement, and high food prices" -- Crisis (IPC Phase 3), multi-driver',
        candidate_taxonomy_cause="COMPOUND (MARKET_SHOCK + DROUGHT)",
    ),
]

FIELDNAMES = [
    "case_id", "zone_codes", "zone_names", "region", "time_period",
    "report_url", "source_quote_or_paraphrase", "candidate_taxonomy_cause",
    "ground_truth_cause", "ground_truth_notes",
]


def main():
    out_path = RQ5_DIR / "labeling_worksheet.csv"
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDNAMES)
        writer.writeheader()
        for case in CASES:
            row = {
                "case_id": case["case_id"],
                "zone_codes": case["zone_codes"],
                "zone_names": case["zone_names"],
                "region": case["region"],
                "time_period": case["time_period"],
                "report_url": case["report_url"],
                "source_quote_or_paraphrase": case["source_quote"],
                "candidate_taxonomy_cause": case["candidate_taxonomy_cause"],
                "ground_truth_cause": "",
                "ground_truth_notes": "",
            }
            writer.writerow(row)
    print(f"wrote {len(CASES)} candidate cases to {out_path}")
    print("\nGROUND TRUTH CATEGORIES to use in ground_truth_cause (must match rule_engine.CATEGORY_NAMES"
          " + COMPOUND): DROUGHT, FLOODING, CONFLICT, MARKET_SHOCK, COMPOUND")


if __name__ == "__main__":
    main()
