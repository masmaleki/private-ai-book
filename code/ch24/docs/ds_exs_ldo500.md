---
doc_id: DS-0002
title: EXS-LDO500-SOT23 datasheet excerpt (fictional)
type: datasheet
manufacturer: Example Semiconductors
mpn: EXS-LDO500-SOT23
visibility: public
version: rev A
date: 2025-11-20
---
# EXS-LDO500-SOT23 500 mA Low-Dropout Regulator

## 1 Features
- Input voltage range: 2.5 V to 16 V
- Output current up to 500 mA
- Dropout voltage 250 mV at 500 mA (typical)
- Fixed outputs: 1.8 V, 3.3 V, 5.0 V
- Package: SOT-23-5

## 6 Electrical Characteristics
| Parameter | Condition | Typ | Max | Unit |
|---|---|---|---|---|
| Input voltage | | | 16 | V |
| Dropout | IOUT = 500 mA | 250 | 400 | mV |
| PSRR | 1 kHz | 65 | | dB |

## 8 Application Notes
An LDO wastes (VIN - VOUT) x IOUT as heat. From a 12 V rail to 3.3 V at
400 mA the regulator dissipates about 3.5 W, which is far too much for
SOT-23. For large input-output differences use a step-down converter such as
EXS-BUCK36-SO8 instead.
