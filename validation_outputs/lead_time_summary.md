# Validation: detection lead time per fault

Synthetic data pending real dataset. Lead time = fault_log timestamp minus the first sustained Warning/Critical crossing.

| unit_id    | subsystem_type   | fault_type       |   severity | fault_timestamp               | first_warning_timestamp   |   lead_time_hours |   lead_time_days | detected_before_fault   |
|:-----------|:-----------------|:-----------------|-----------:|:------------------------------|:--------------------------|------------------:|-----------------:|:------------------------|
| DOOR_04_2  | door             | motor_wear       |          4 | 2026-08-12 01:02:30.671013718 | 2026-08-01 17:03:12       |            247.99 |            10.33 | True                    |
| BOGIE_06_2 | bogie            | wheel_flat       |          4 | 2026-08-13 07:45:23.320651343 | 2026-08-09 07:14:20       |             96.52 |             4.02 | True                    |
| DOOR_03_2  | door             | seal_degradation |          3 | 2026-08-17 10:15:04.282844726 | 2026-08-15 10:01:09       |             48.23 |             2.01 | True                    |
| BOGIE_01_2 | bogie            | wheel_flat       |          4 | 2026-08-17 13:13:09.872920194 | 2026-08-07 01:49:53       |            251.39 |            10.47 | True                    |
| DOOR_06_1  | door             | seal_degradation |          4 | 2026-08-19 03:09:12.244117788 | 2026-08-14 16:28:44       |            106.67 |             4.44 | True                    |
| BOGIE_04_1 | bogie            | bearing_failure  |          3 | 2026-08-21 19:47:53.479570924 | 2026-08-15 13:14:54       |            150.55 |             6.27 | True                    |
| DOOR_01_1  | door             | seal_degradation |          5 | 2026-08-28 14:15:38.977031873 | 2026-08-25 06:43:32       |             79.54 |             3.31 | True                    |
| BOGIE_02_1 | bogie            | suspension_wear  |          5 | 2026-09-02 10:40:02.686019398 | 2026-08-23 09:32:32       |            241.13 |            10.05 | True                    |

**Detected before fault: 8/8**

Mean lead time (detected only): 152.8 hours
