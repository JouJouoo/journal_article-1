@echo off
REM ============================================
REM DP-LCRL-RL 环境激活脚本
REM 使用方法: 双击运行，或在命令行执行
REM ============================================

echo Activating DP-LCRL environment...
call C:\ProgramData\anaconda3\Scripts\activate.bat DP-LCRL
echo Environment ready!
echo.
echo 可用命令：
echo   python --version           检查Python版本
echo   python -c "import torch; print(torch.cuda.is_available())"  检查GPU
echo.
echo 运行训练:
echo   python -m dp_lcrl_rl.scripts.train.train_paper_mat --experiment_name test --num_agents 10 --min_agents 5 --n_rollout_threads 2 --num_env_steps 24000 --use_eval True
echo.
echo 运行评估:
echo   python -m dp_lcrl_rl.scripts.eval.eval_paper_mat --experiment_name test --model_dir runs/test/models
echo.

cmd /k
