# Validation: detection lead time per fault

Synthetic data pending real dataset. Lead time = fault_log timestamp minus the first sustained Warning/Critical crossing.

| unit_id    | subsystem_type   | fault_type              |   severity | fault_timestamp               | first_warning_timestamp   |   lead_time_hours |   lead_time_days | detected_before_fault   |
|:-----------|:-----------------|:------------------------|-----------:|:------------------------------|:--------------------------|------------------:|-----------------:|:------------------------|
| CAR_05_1   | car              | hvac_compressor_wear    |          5 | 2026-08-07 23:34:19.834986153 | 2026-08-04 06:10:17       |             89.4  |             3.73 | True                    |
| BOGIE_04_1 | bogie            | brake_pad_wear          |          4 | 2026-08-11 00:05:40.949154778 | 2026-08-08 23:38:03       |             48.46 |             2.02 | True                    |
| DOOR_06_1  | door             | track_misalignment      |          4 | 2026-08-14 23:39:10.759584786 | 2026-08-13 12:47:14       |             34.87 |             1.45 | True                    |
| CAR_04_1   | car              | comms_antenna_fault     |          4 | 2026-08-17 09:21:30.358717753 | 2026-08-12 02:20:52       |            127.01 |             5.29 | True                    |
| DOOR_01_1  | door             | seal_degradation        |          4 | 2026-08-21 02:40:45.538894961 | 2026-08-13 17:11:12       |            177.49 |             7.4  | True                    |
| BOGIE_06_2 | bogie            | bearing_failure         |          3 | 2026-08-21 19:06:55.424931950 | 2026-08-16 10:30:26       |            128.61 |             5.36 | True                    |
| DOOR_04_2  | door             | seal_degradation        |          4 | 2026-08-24 03:28:49.569006547 | 2026-08-20 02:32:16       |             96.94 |             4.04 | True                    |
| DOOR_03_2  | door             | track_misalignment      |          4 | 2026-08-25 08:32:36.255127765 | 2026-08-17 05:14:14       |            195.31 |             8.14 | True                    |
| BOGIE_02_1 | bogie            | wheel_flat              |          4 | 2026-08-27 00:47:17.738161508 | 2026-08-22 11:28:48       |            109.31 |             4.55 | True                    |
| CAR_04_2   | car              | lighting_driver_failure |          5 | 2026-09-02 09:27:37.864492414 | 2026-08-24 03:40:32       |            221.78 |             9.24 | True                    |
| BOGIE_01_2 | bogie            | wheel_flat              |          4 | 2026-09-03 21:43:44.694172932 | 2026-09-03 20:40:37       |              1.05 |             0.04 | True                    |

**Detected before fault: 11/11**

Mean lead time (detected only): 111.8 hours
