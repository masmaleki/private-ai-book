---
doc_id: AN-0101
title: Application note - PCB layout for 3 A buck converters (fictional)
type: app_note
manufacturer: Example Semiconductors
visibility: public
version: 1.2
date: 2025-09-10
---
# AN-0101 Layout guidelines for step-down converters

## Hot loop
Keep the input capacitor, the switch node and the ground return as close as
possible. The high di/dt loop radiates noise if it is large.

## Thermal design
For SO-8 packages with exposed pad, place 4 to 9 thermal vias of 0.3 mm
under the pad and connect them to a ground plane. At high ambient
temperature the copper area decides how much current the converter can
deliver; see the derating curve in each datasheet.

## Feedback trace
Route the feedback trace away from the inductor and the switch node.
