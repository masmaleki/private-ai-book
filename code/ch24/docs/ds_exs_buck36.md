---
doc_id: DS-0001
title: EXS-BUCK36-SO8 datasheet excerpt (fictional)
type: datasheet
manufacturer: Example Semiconductors
mpn: EXS-BUCK36-SO8
visibility: public
version: rev C
date: 2026-03-02
---
# EXS-BUCK36-SO8 3 A Step-Down Converter

## 1 Features
- Wide input voltage range: 4.5 V to 36 V
- Continuous output current up to 3 A at TA = 25 °C
- Adjustable output voltage from 0.8 V to 24 V
- Fixed switching frequency 500 kHz
- Package: SO-8 with exposed thermal pad

## 2 Applications
Industrial 24 V rails, sensor modules, PLC I/O boards, telecom point-of-load.

## 6 Electrical Characteristics
| Parameter | Condition | Min | Typ | Max | Unit |
|---|---|---|---|---|---|
| Input voltage VIN | | 4.5 | | 36 | V |
| Quiescent current | no load, VIN = 24 V | | 85 | 120 | µA |
| Output current IOUT | TA = 25 °C | | | 3.0 | A |
| Output current IOUT | TA = 105 °C | | | 2.1 | A |
| Efficiency | VIN = 24 V, VOUT = 5 V, 1 A | | 91 | | % |

## 7 Thermal Derating
Above 85 °C ambient the maximum output current decreases linearly from 3.0 A
to 2.1 A at 105 °C. Operation above 125 °C junction temperature is not
recommended. Use at least 4 thermal vias under the exposed pad.

## 9 Absolute Maximum Ratings
VIN pin: -0.3 V to 40 V. Stresses beyond these ratings may cause permanent
damage. The recommended operating maximum is 36 V.
