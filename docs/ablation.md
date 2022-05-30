
 - ImageNet-100
 
|Method|Status| 100 epochs | 200 epochs | 300 epochs|
|:-----|:-----:|:-----:|:----:|:----:|
|BASELINE| [x] |82.76/96.22|88.40/97.78|89.68/97.98|
|BASELINE + `ape`| [x] |83.08/96.18|88.48/97.76|89.96/98.04|
|+ `reld` (`l1`, last layer)| [x] |83.38/96.54|88.64/97.86|90.04/98.22|
|+ `reld` (`l1`, last layer) + `ape`| [x] |83.52/96.58|88.90/97.72|90.12/97.94|
|+ `reld` (`l1`, all layers)| [x] |83.18/96.38|88.26/97.66|89.52/98.24|
|+ `reld` (`l1`, all layers) + `ape`| [x] |83.74/96.44|89.02/98.06|90.08/98.22|
|+ `reld` (`l1`, last layer)`(NM)`| [x] |82.60/96.02|
|+ `reld` (`l1`, all layers)`(NM)`| [x] |82.92/96.12|
|+ `reld` (`l1`, all layers)`(NM)` + `ape`| [x] |83.26/96.38|
|+ `reld` (`l1^`, last layers) - `rpe`|
|+ `reld` (`l1^`, last layers)|
|+ `reld` (`l1^`, last layer) + `ape`|
|+ `reld` (`l1^`, all layers) - `rpe`|
|+ `reld` (`l1^`, all layers)|[x]|83.20/96.36|
|+ `reld` (`l1^`, all layers) + `ape`|
|+ `reld` (`ce`, last layer) - `rpe`|
|+ `reld` (`ce`, last layer)|
|+ `reld` (`ce`, last layer) + `ape`|
|+ `reld` (`ce`, all layers) - `rpe`|
|+ `reld` (`ce`, all layers)|...|
|+ `reld` (`ce`, all layers) + `ape`|...|
|+ `reld` (`ce^`, last layer) - `rpe`|[x]|83.08/96.62|
|+ `reld` (`ce^`, last layer)|[x]|83.52/96.18|
|+ `reld` (`ce^`, last layer) + `ape`|[x]|83.10/96.40|
|+ `reld` (`ce^`, all layers) - `rpe`|[x]|81.56/95.44|
|+ `reld` (`ce^`, all layers)|[x]|82.16/95.86|
|+ `reld` (`ce^`, all layers) + `ape`|[x]|82.24/96.12|
|+ `reld` (`cbr`, last layer) - `rpe`|
|+ `reld` (`cbr`, last layer)|
|+ `reld` (`cbr`, last layer) + `ape`|
|+ `reld` (`cbr`, all layers) - `rpe`|
|+ `reld` (`cbr`, all layers)|...|
|+ `reld` (`cbr`, all layers) + `ape`|...|
