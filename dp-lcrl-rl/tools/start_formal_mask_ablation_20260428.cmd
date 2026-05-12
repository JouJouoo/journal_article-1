@echo off
setlocal
cd /d "%~dp0\.."

if not exist "reports\formal_mask_ablation_20260428" mkdir "reports\formal_mask_ablation_20260428"
if not exist "reports\formal_mask_ablation_20260428\logs" mkdir "reports\formal_mask_ablation_20260428\logs"
if not exist "tmp_eval_runtime" mkdir "tmp_eval_runtime"

set "KMP_DUPLICATE_LIB_OK=TRUE"
set "OMP_NUM_THREADS=1"
set "MKL_NUM_THREADS=1"
set "OPENBLAS_NUM_THREADS=1"
set "NUMEXPR_NUM_THREADS=1"
set "WANDB_DISABLED=true"
set "WANDB_MODE=disabled"
set "DP_LCRL_DISABLE_TENSORBOARD=1"
set "MPLBACKEND=Agg"
set "TMP=%CD%\tmp_eval_runtime"
set "TEMP=%CD%\tmp_eval_runtime"
set "PYTHONIOENCODING=utf-8"

"C:\Users\23034\.conda\envs\mat\python.exe" -m dp_lcrl_rl.scripts.run_formal_mask_ablation_20260428 > "reports\formal_mask_ablation_20260428\launcher_stdout.log" 2> "reports\formal_mask_ablation_20260428\launcher_stderr.log"
endlocal
