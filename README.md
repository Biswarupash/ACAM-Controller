# ACAM-Controller
Here, we include all the codes and data associated with this project.


ACAM Controller - adaptive camber wing controller
=================================================
Implements Steps 1-14 of "ACAM Controller: Coder's Guide" (Lucas Fawcett).

DATA MODE (the key switch)
--------------------------
The user states what kind of wing data is being used, either in the config
file ("data_mode": ...) or on the command line (--data-mode ...):

    "random" / "mock"   -> made-up placeholder polars (Part 1). Results only
                           prove the code works; never use them for conclusions.
    "genuine" / "real"  -> real XFOIL CSV (Part 2). The file is loaded and
                           validated, and real wing constants are required.

If neither is given, the program asks interactively.

OUTPUT
------
  acam_log.csv               every control step (inputs, CL_req, camber, flags)
  camber_vs_time.png         camber command vs time, phases shaded
  airfoil_shapes.png         wing section shape at the camber used in each phase
  airfoil_shape_coords.csv   x / camber-line / upper / lower surface coordinates
  checklist.csv              Step 9 test results (pass / fail)
  drag_comparison.csv        adaptive vs fixed-camber drag + stall violations
  polars_check.png           (genuine mode) cd vs cl per camber for review
  acam_ai_model.joblib       (if AI enabled) trained network + scaler

USAGE
-----
  python acam_controller.py --data-mode random
  
  python acam_controller.py --data-mode genuine --xfoil xfoil_polars.csv --S 16 --b 10 --W 10000
  
  python acam_controller.py --config acam_config.json --ai
