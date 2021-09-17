#!/bin/sh
set -eu

# Official code of "Keisuke Izumiya and Edgar Simo-Serra, Inventory Management with Attention-Based Meta Actions, IEEE Conference on Games (CoG), 2021."
#    Copyright (C) 2021 Keisuke Izumiya
#
#    This program is free software: you can redistribute it and/or modify
#    it under the terms of the GNU General Public License as published by
#    the Free Software Foundation, either version 3 of the License, or
#    (at your option) any later version.
#
#    This program is distributed in the hope that it will be useful,
#    but WITHOUT ANY WARRANTY; without even the implied warranty of
#    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
#    GNU General Public License for more details.
#
#    You should have received a copy of the GNU General Public License
#    along with this program.  If not, see <https://www.gnu.org/licenses/>.

CMD="python -m nle.agent.polyhydra"

NO_INV="model.inv.model=none model.policy.model=baseline"
NO_ACTION="model.action.model=none"
REDUCED_ACTION="env.reduced_action=true"

MON="env.character=mon-hum-neu-mal"
VAL="env.character=val-dwa-law-fem"
TOU="env.character=tou-hum-neu-fem"

# Monk
$CMD $MON $NO_INV $NO_ACTION $REDUCED_ACTION  # [28]
$CMD $MON $NO_INV $NO_ACTION                  # Baseline
$CMD $MON $NO_INV                             # Ours w/o Inventory
$CMD $MON         $NO_ACTION                  # Ours w/o Action Recursion
$CMD $MON                                     # Ours

# Valkyrie
$CMD $VAL $NO_INV $NO_ACTION $REDUCED_ACTION  # [28]
$CMD $VAL $NO_INV $NO_ACTION                  # Baseline
$CMD $VAL $NO_INV                             # Ours w/o Inventory
$CMD $VAL         $NO_ACTION                  # Ours w/o Action Recursion
$CMD $VAL                                     # Ours

# Tourist
$CMD $TOU $NO_INV $NO_ACTION $REDUCED_ACTION  # [28]
$CMD $TOU $NO_INV $NO_ACTION                  # Baseline
$CMD $TOU $NO_INV                             # Ours w/o Inventory
$CMD $TOU         $NO_ACTION                  # Ours w/o Action Recursion
$CMD $TOU                                     # Ours
